"""
Object Tracking Pipeline
========================
Video → YOLO Segmentation → ByteTrack → Crop → DINOv2 Embedding
→ FAISS Re-ID → Registry Update → SQLite + JSON + FAISS index + Top Crops

Design decisions (confirmed):
  - Embedding        : DINOv2 (facebook/dinov2-small via HuggingFace)
  - Re-ID            : FAISS cosine similarity, threshold = 0.25
  - Top crops        : 6 per object, ranked by bounding-box pixel area
  - Frame gap        : new range entry if gap > 5 frames
  - Re-ID auth       : FAISS is sole authority
  - Min frames guard : object is only committed to registry after being
                       seen in >= min_frames_to_commit (5) frames.
                       Until then it lives in a "pending buffer" and is
                       silently discarded if it never reaches the threshold.

Output layout (one folder per video):
  <output_root>/
  ├── <video_stem>.db          SQLite database
  ├── <video_stem>.json        Full metadata (one file per video)
  ├── <video_stem>.index       FAISS FlatIP index (L2-normalised → cosine)
  ├── <video_stem>.npz         id→embedding map (numpy)
  └── crops/
      └── obj_<global_id>/
          └── frame_<n>_area_<px>.jpg

─────────────────────────────────────────────────────────────────────────────
HOW TO ACCESS THE DATABASE
─────────────────────────────────────────────────────────────────────────────

Schema:
    objects       (global_id PK, class_label, faiss_row, embed_count)
    frame_ranges  (id PK, global_id FK, start_frame, end_frame)
    crops         (id PK, global_id FK, frame_idx, area_px, file_path)

── Python ───────────────────────────────────────────────────────────────────

    import sqlite3

    con = sqlite3.connect("output/sarthak/sarthak.db")
    con.row_factory = sqlite3.Row          # access columns by name

    # 1. List all tracked objects
    for row in con.execute("SELECT * FROM objects"):
        print(dict(row))

    # 2. All frame ranges for object with global_id = 3
    for row in con.execute(
        "SELECT start_frame, end_frame FROM frame_ranges WHERE global_id = ?", (3,)
    ):
        print(row["start_frame"], "→", row["end_frame"])

    # 3. Top crops for object 3 (best first)
    for row in con.execute(
        "SELECT frame_idx, area_px, file_path FROM crops "
        "WHERE global_id = ? ORDER BY area_px DESC", (3,)
    ):
        print(row["file_path"])

    # 4. All objects visible between frame 100 and 300
    for row in con.execute(""
        SELECT DISTINCT o.global_id, o.class_label
        FROM objects o
        JOIN frame_ranges r ON o.global_id = r.global_id
        WHERE r.start_frame <= 300 AND r.end_frame >= 100
    ""):
        print(dict(row))

    # 5. Objects ranked by total frames they appeared in
    for row in con.execute(""
        SELECT o.global_id, o.class_label,
               SUM(r.end_frame - r.start_frame + 1) AS total_frames
        FROM objects o
        JOIN frame_ranges r ON o.global_id = r.global_id
        GROUP BY o.global_id
        ORDER BY total_frames DESC
    ""):
        print(dict(row))

    con.close()

── CLI (sqlite3 shell) ──────────────────────────────────────────────────────

    sqlite3 output/sarthak/sarthak.db

    .headers on
    .mode column
    SELECT * FROM objects;
    SELECT * FROM frame_ranges WHERE global_id = 3;
    SELECT * FROM crops WHERE global_id = 3 ORDER BY area_px DESC;

── JSON ─────────────────────────────────────────────────────────────────────

    import json
    with open("output/sarthak/sarthak.json") as f:
        meta = json.load(f)
    for obj in meta["objects"]:
        print(obj["global_id"], obj["class_label"], obj["frame_ranges"])

── NPZ (embeddings) ─────────────────────────────────────────────────────────

    import numpy as np
    data = np.load("output/sarthak/sarthak.npz")
    ids        = data["global_ids"]    # shape (N,)
    embeddings = data["embeddings"]    # shape (N, D)

── FAISS index ──────────────────────────────────────────────────────────────

    import faiss, numpy as np
    index = faiss.read_index("output/sarthak/sarthak.index")
    query = embeddings[0:1]                    # shape (1, D), unit-normalised
    scores, rows = index.search(query, 5)      # top-5 nearest objects
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import faiss
import numpy as np
import supervision as sv
import torch
from PIL import Image
from ultralytics import YOLO
from trackers import ByteTrackTracker


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

@dataclass
class PipelineConfig:
    video_path:   str   = "data/sarthak.mp4"
    output_root:  str   = "output"
    yolo_model:   str   = "models/yolo26s.pt"

    faiss_threshold:      float = 0.25
    top_crops:            int   = 6
    frame_gap_tolerance:  int   = 5
    min_frames_to_commit: int   = 5     # ★ objects seen < N frames are discarded

    device:     str   = "cpu"
    dino_model: str   = "facebook/dinov2-small"
    yolo_conf:  float = 0.3
    yolo_iou:   float = 0.5

    show_preview: bool = True
    log_level:    str  = "INFO"


# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class CropRecord:
    frame_idx: int
    area_px:   int
    file_path: str


@dataclass
class ObjectRecord:
    global_id:   int
    class_label: str
    frame_ranges: List[Tuple[int, int]] = field(default_factory=list)

    _open_range_start: int = -1
    _last_seen:        int = -1

    top_crops:       List[CropRecord]    = field(default_factory=list)
    embedding:       Optional[np.ndarray] = None
    embedding_count: int = 0
    faiss_row:       int = -1

    def update_range(self, frame_idx: int, gap_tolerance: int):
        if self._open_range_start == -1:
            self._open_range_start = frame_idx
        elif frame_idx - self._last_seen > gap_tolerance:
            self.frame_ranges.append((self._open_range_start, self._last_seen))
            self._open_range_start = frame_idx
        self._last_seen = frame_idx

    def close_range(self):
        if self._open_range_start != -1 and self._last_seen != -1:
            self.frame_ranges.append((self._open_range_start, self._last_seen))
            self._open_range_start = -1
            self._last_seen        = -1

    def add_crop(self, crop: CropRecord, max_crops: int):
        self.top_crops.append(crop)
        self.top_crops.sort(key=lambda c: c.area_px, reverse=True)
        if len(self.top_crops) > max_crops:
            self.top_crops = self.top_crops[:max_crops]

    def update_embedding(self, new_vec: np.ndarray):
        if self.embedding is None:
            self.embedding = new_vec.copy()
        else:
            n = self.embedding_count
            self.embedding = (self.embedding * n + new_vec) / (n + 1)
            norm = np.linalg.norm(self.embedding)
            if norm > 0:
                self.embedding /= norm
        self.embedding_count += 1


# ──────────────────────────────────────────────
# PENDING BUFFER  (min-frames guard)
# ──────────────────────────────────────────────

@dataclass
class PendingObject:
    """Temporary holding area — promoted only after min_frames sightings."""
    class_label:          str
    frames_seen:          int                      = 0
    buffered_crops:       List[Tuple]              = field(default_factory=list)
    buffered_embeddings:  List[np.ndarray]         = field(default_factory=list)
    first_frame:          int                      = -1
    last_frame:           int                      = -1


class PendingBuffer:
    def __init__(self, min_frames: int):
        self.min_frames = min_frames
        self._buf: Dict[str, PendingObject] = {}

    def observe(self, key: str, class_label: str,
                frame_idx: int, area_px: int,
                bgr_crop: np.ndarray, embedding: np.ndarray) -> bool:
        """Record a sighting. Returns True when the object is ready to promote."""
        if key not in self._buf:
            self._buf[key] = PendingObject(
                class_label=class_label,
                first_frame=frame_idx,
            )
        p = self._buf[key]
        p.frames_seen += 1
        p.last_frame   = frame_idx
        p.buffered_crops.append((frame_idx, area_px, bgr_crop.copy()))
        p.buffered_embeddings.append(embedding.copy())
        return p.frames_seen >= self.min_frames

    def pop(self, key: str) -> Optional[PendingObject]:
        return self._buf.pop(key, None)

    def pending_keys(self):
        return list(self._buf.keys())


# ──────────────────────────────────────────────
# DINO EMBEDDER
# ──────────────────────────────────────────────

class DinoEmbedder:
    def __init__(self, model_name: str = "facebook/dinov2-small", device: str = "cpu"):
        self.device = device
        logging.info(f"Loading DINOv2: {model_name} on {device}")
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def embed(self, bgr_crop: np.ndarray) -> np.ndarray:
        rgb    = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb),
                                return_tensors="pt").to(self.device)
        feat   = self.model(**inputs).last_hidden_state[:, 0, :]
        feat   = feat.cpu().numpy().astype(np.float32)[0]
        norm   = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        return feat


# ──────────────────────────────────────────────
# FAISS REGISTRY
# ──────────────────────────────────────────────

class FAISSRegistry:
    def __init__(self, dim: int, threshold: float):
        self.dim       = dim
        self.threshold = threshold
        self.index     = faiss.IndexFlatIP(dim)
        self.row_to_gid: List[int] = []

    def search(self, vec: np.ndarray) -> Optional[int]:
        if self.index.ntotal == 0:
            return None
        q = vec.reshape(1, -1).astype(np.float32)
        scores, rows = self.index.search(q, 1)
        cos_dist = 1.0 - float(scores[0, 0])
        if cos_dist <= self.threshold:
            return self.row_to_gid[int(rows[0, 0])]
        return None

    def add(self, vec: np.ndarray, global_id: int) -> int:
        self.index.add(vec.reshape(1, -1).astype(np.float32))
        row = len(self.row_to_gid)
        self.row_to_gid.append(global_id)
        return row

    def update_row(self, row: int, vec: np.ndarray):
        if self.index.ntotal == 0:
            return
        all_vecs = np.zeros((self.index.ntotal, self.dim), dtype=np.float32)
        for i in range(self.index.ntotal):
            self.index.reconstruct(i, all_vecs[i])
        all_vecs[row] = vec.astype(np.float32)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(all_vecs)

    def save(self, path: str):
        faiss.write_index(self.index, path)


# ──────────────────────────────────────────────
# STORAGE
# ──────────────────────────────────────────────

class StorageManager:
    def __init__(self, output_dir: Path, video_stem: str):
        self.output_dir = output_dir
        self.crops_dir  = output_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(output_dir / f"{video_stem}.db"))
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS objects (
                global_id   INTEGER PRIMARY KEY,
                class_label TEXT,
                faiss_row   INTEGER,
                embed_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS frame_ranges (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                global_id   INTEGER,
                start_frame INTEGER,
                end_frame   INTEGER,
                FOREIGN KEY(global_id) REFERENCES objects(global_id)
            );
            CREATE TABLE IF NOT EXISTS crops (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                global_id   INTEGER,
                frame_idx   INTEGER,
                area_px     INTEGER,
                file_path   TEXT,
                FOREIGN KEY(global_id) REFERENCES objects(global_id)
            );
        """)
        self.conn.commit()

    def upsert_object(self, obj: ObjectRecord):
        self.conn.execute("""
            INSERT INTO objects(global_id, class_label, faiss_row, embed_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(global_id) DO UPDATE SET
                faiss_row=excluded.faiss_row,
                embed_count=excluded.embed_count
        """, (obj.global_id, obj.class_label, obj.faiss_row, obj.embedding_count))
        self.conn.commit()

    def save_frame_ranges(self, obj: ObjectRecord):
        self.conn.execute("DELETE FROM frame_ranges WHERE global_id=?", (obj.global_id,))
        self.conn.executemany(
            "INSERT INTO frame_ranges(global_id, start_frame, end_frame) VALUES (?,?,?)",
            [(obj.global_id, s, e) for s, e in obj.frame_ranges]
        )
        self.conn.commit()

    def save_crops(self, obj: ObjectRecord):
        self.conn.execute("DELETE FROM crops WHERE global_id=?", (obj.global_id,))
        self.conn.executemany(
            "INSERT INTO crops(global_id, frame_idx, area_px, file_path) VALUES (?,?,?,?)",
            [(obj.global_id, c.frame_idx, c.area_px, c.file_path) for c in obj.top_crops]
        )
        self.conn.commit()

    def write_crop_image(self, global_id: int, frame_idx: int,
                         area_px: int, bgr_crop: np.ndarray) -> str:
        obj_dir = self.crops_dir / f"obj_{global_id:04d}"
        obj_dir.mkdir(exist_ok=True)
        fpath = obj_dir / f"frame_{frame_idx:06d}_area_{area_px}.jpg"
        cv2.imwrite(str(fpath), bgr_crop)
        return str(fpath.relative_to(self.output_dir))

    def export_json(self, registry: Dict[int, ObjectRecord], video_stem: str):
        out = {
            "video": video_stem,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_objects": len(registry),
            "objects": [
                {
                    "global_id":       gid,
                    "class_label":     obj.class_label,
                    "frame_ranges":    obj.frame_ranges,
                    "embedding_count": obj.embedding_count,
                    "faiss_row":       obj.faiss_row,
                    "top_crops": [
                        {"frame": c.frame_idx, "area_px": c.area_px, "path": c.file_path}
                        for c in obj.top_crops
                    ]
                }
                for gid, obj in sorted(registry.items())
            ]
        }
        json_path = self.output_dir / f"{video_stem}.json"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        logging.info(f"JSON saved  → {json_path}")

    def export_embeddings_npz(self, registry: Dict[int, ObjectRecord], video_stem: str):
        ids  = [gid for gid, obj in registry.items() if obj.embedding is not None]
        vecs = [obj.embedding for obj in registry.values() if obj.embedding is not None]
        if vecs:
            npz_path = self.output_dir / f"{video_stem}.npz"
            np.savez(str(npz_path),
                     global_ids=np.array(ids, dtype=np.int32),
                     embeddings=np.stack(vecs, axis=0))
            logging.info(f"NPZ saved   → {npz_path}")

    def close(self):
        self.conn.close()


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

class ObjectPipeline:

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        logging.basicConfig(
            level=getattr(logging, cfg.log_level),
            format="%(asctime)s [%(levelname)s] %(message)s"
        )

        logging.info("Loading YOLO …")
        self.yolo     = YOLO(cfg.yolo_model)
        self.tracker  = ByteTrackTracker()
        self.embedder = DinoEmbedder(cfg.dino_model, cfg.device)

        self.faiss_reg = FAISSRegistry(
            dim=self.embedder.dim,
            threshold=cfg.faiss_threshold
        )

        self.registry:    Dict[int, ObjectRecord] = {}
        self._next_gid:   int                     = 0
        self._tid_to_gid: Dict[str, int]          = {}   # tracker_id → global_id

        # ★ Objects must be seen >= min_frames_to_commit before entering registry
        self.pending = PendingBuffer(min_frames=cfg.min_frames_to_commit)

        video_stem      = Path(cfg.video_path).stem
        self.video_stem = video_stem
        self.output_dir = Path(cfg.output_root) / video_stem
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.storage   = StorageManager(self.output_dir, video_stem)
        self.box_ann   = sv.BoxAnnotator()
        self.label_ann = sv.LabelAnnotator()

    # ── Helpers ─────────────────────────────────

    def _new_gid(self) -> int:
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def _crop(self, frame: np.ndarray, xyxy: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, xyxy)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _promote(self, pending_key: str, matched_gid: Optional[int]) -> int:
        """
        Flush PendingObject into the real registry.
        Merges into matched_gid if provided, else creates a new ObjectRecord.
        All buffered crops and embeddings are replayed in order.
        """
        p = self.pending.pop(pending_key)
        if p is None:
            return -1

        if matched_gid is not None and matched_gid in self.registry:
            obj = self.registry[matched_gid]
        else:
            gid = self._new_gid()
            obj = ObjectRecord(global_id=gid, class_label=p.class_label)
            self.registry[gid] = obj

        # Replay buffered data
        for emb in p.buffered_embeddings:
            obj.update_embedding(emb)

        for (frame_idx, area_px, bgr_crop) in p.buffered_crops:
            obj.update_range(frame_idx, self.cfg.frame_gap_tolerance)
            rel_path = self.storage.write_crop_image(
                obj.global_id, frame_idx, area_px, bgr_crop
            )
            obj.add_crop(
                CropRecord(frame_idx=frame_idx, area_px=area_px, file_path=rel_path),
                self.cfg.top_crops
            )

        # Register in FAISS (or update existing row)
        if obj.faiss_row == -1:
            obj.faiss_row = self.faiss_reg.add(obj.embedding, obj.global_id)
        else:
            self.faiss_reg.update_row(obj.faiss_row, obj.embedding)

        self.storage.upsert_object(obj)
        logging.debug(
            f"Promoted  G{obj.global_id} | class={p.class_label} | "
            f"buffered={p.frames_seen} frames"
        )
        return obj.global_id

    # ── Per-detection ────────────────────────────

    def _process_detection(self, frame: np.ndarray, frame_idx: int,
                            tracker_id: int, xyxy: np.ndarray,
                            class_label: str) -> str:
        """
        Returns a display label:
          G<n>  → committed global object
          P<n>  → still pending (not yet shown in registry)
          ?     → bad crop
        """
        crop = self._crop(frame, xyxy)
        if crop is None or crop.size == 0:
            return "?"

        area_px = (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))
        vec     = self.embedder.embed(crop)
        tid_key = f"tid_{tracker_id}"

        # ── Already committed tracker ────────────────────────────────────
        if tid_key in self._tid_to_gid:
            gid = self._tid_to_gid[tid_key]
            obj = self.registry[gid]
            obj.update_embedding(vec)
            self.faiss_reg.update_row(obj.faiss_row, obj.embedding)
            obj.update_range(frame_idx, self.cfg.frame_gap_tolerance)
            rel = self.storage.write_crop_image(obj.global_id, frame_idx, area_px, crop)
            obj.add_crop(CropRecord(frame_idx, area_px, rel), self.cfg.top_crops)
            self.storage.upsert_object(obj)
            return f"G{gid}"

        # ── Pending: accumulate frames ───────────────────────────────────
        pending_key  = f"pending_{tracker_id}"
        matched_gid  = self.faiss_reg.search(vec)   # check against committed objects

        ready = self.pending.observe(
            key=pending_key,
            class_label=class_label,
            frame_idx=frame_idx,
            area_px=area_px,
            bgr_crop=crop,
            embedding=vec,
        )

        if not ready:
            return f"P{tracker_id}"   # still accumulating

        # ── Threshold reached: promote to registry ───────────────────────
        gid = self._promote(pending_key, matched_gid)
        if gid == -1:
            return "?"

        self._tid_to_gid[tid_key] = gid
        return f"G{gid}"

    # ── Main loop ────────────────────────────────

    def run(self):
        self._tid_to_gid = {}

        cap = cv2.VideoCapture(self.cfg.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {self.cfg.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        logging.info(
            f"Video: {self.cfg.video_path} | "
            f"Frames: {total_frames} | FPS: {fps:.1f} | "
            f"Min-frames guard: {self.cfg.min_frames_to_commit}"
        )

        frame_idx = 0
        t0        = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result     = self.yolo(frame, conf=self.cfg.yolo_conf,
                                   iou=self.cfg.yolo_iou, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            if len(detections) > 0:
                detections = self.tracker.update(detections)

                labels = []
                for xyxy, tracker_id, class_id in zip(
                    detections.xyxy,
                    detections.tracker_id,
                    detections.class_id
                ):
                    if tracker_id is None:
                        labels.append("?")
                        continue
                    lbl = self._process_detection(
                        frame, frame_idx, int(tracker_id),
                        xyxy, result.names.get(int(class_id), str(class_id))
                    )
                    labels.append(lbl)

                if self.cfg.show_preview:
                    ann = self.box_ann.annotate(frame.copy(), detections=detections)
                    ann = self.label_ann.annotate(ann, detections=detections, labels=labels)
                    cv2.imshow("Pipeline Preview", ann)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                logging.info(
                    f"Frame {frame_idx}/{total_frames} | "
                    f"Committed: {len(self.registry)} | "
                    f"Pending: {len(self.pending._buf)} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

        cap.release()
        cv2.destroyAllWindows()
        self._finalise()

    # ── Finalise ─────────────────────────────────

    def _finalise(self):
        logging.info("Finalising …")

        discarded = len(self.pending._buf)
        if discarded:
            logging.info(
                f"Discarding {discarded} object(s) that never reached "
                f"{self.cfg.min_frames_to_commit} frames."
            )

        for obj in self.registry.values():
            obj.close_range()
            self.storage.save_frame_ranges(obj)
            self.storage.save_crops(obj)

        faiss_path = str(self.output_dir / f"{self.video_stem}.index")
        self.faiss_reg.save(faiss_path)
        logging.info(f"FAISS index → {faiss_path}")

        self.storage.export_embeddings_npz(self.registry, self.video_stem)
        self.storage.export_json(self.registry, self.video_stem)
        self.storage.close()

        logging.info(
            f"\n{'='*52}\n"
            f"  Done.\n"
            f"  Committed objects           : {len(self.registry)}\n"
            f"  Discarded (< {self.cfg.min_frames_to_commit} frames) : {discarded}\n"
            f"  Output → {self.output_dir}\n"
            f"{'='*52}"
        )


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cfg = PipelineConfig(
        video_path   = "data/sarthak.mp4",
        output_root  = "output",
        yolo_model   = "models/yolo26s.pt",

        faiss_threshold      = 0.25,
        top_crops            = 6,
        frame_gap_tolerance  = 5,
        min_frames_to_commit = 5,    # ★ discard flickers / false positives

        device       = "cpu",        # swap to "cuda" if GPU available
        dino_model   = "facebook/dinov2-small",
        yolo_conf    = 0.3,
        yolo_iou     = 0.5,
        show_preview = True,
        log_level    = "INFO",
    )

    pipeline = ObjectPipeline(cfg)
    pipeline.run()