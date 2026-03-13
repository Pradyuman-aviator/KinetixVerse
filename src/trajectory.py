import cv2
import torch
import faiss
import numpy as np
import open3d as o3d
import trimesh
import logging

from tqdm import tqdm
from kornia.feature import LoFTR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("MeshPoseTracker")


class MeshPoseTracker:

    def __init__(self, mesh_path, width=640, height=480):

        self.W = width
        self.H = height

        self.fx = self.fy = 700
        self.cx = width / 2
        self.cy = height / 2

        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info("Initializing MeshPoseTracker")

        self._load_mesh(mesh_path)
        self._setup_renderer()

        self.loftr = LoFTR(pretrained="outdoor").to(self.device).eval()

        self.synthetic_views = []
        self.faiss_index = None
        self.trajectory = []


    # --------------------------------------------------
    # Load mesh
    # --------------------------------------------------

    def _load_mesh(self, mesh_path):

        logger.info("Loading mesh")

        mesh = trimesh.load(mesh_path)

        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)

        vertices -= vertices.mean(axis=0)
        vertices /= np.max(np.linalg.norm(vertices, axis=1))

        self.mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces)
        )

        self.mesh.compute_vertex_normals()


    # --------------------------------------------------
    # Renderer
    # --------------------------------------------------

    def _setup_renderer(self):

        logger.info("Setting up renderer")

        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.W, self.H
        )

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"

        self.renderer.scene.add_geometry("mesh", self.mesh, material)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.W, self.H, self.fx, self.fy, self.cx, self.cy
        )

        extrinsic = np.eye(4)
        extrinsic[2, 3] = 2.2

        self.renderer.setup_camera(intrinsics, extrinsic)

        self.renderer.scene.scene.set_indirect_light_intensity(60000)
        self.renderer.scene.set_background([0.9, 0.9, 0.9, 1])


    # --------------------------------------------------
    # Camera pose sampling
    # --------------------------------------------------

    def sample_poses(self, n=200, radius_range=(1.8, 3.0)):

        poses = []

        for _ in range(n):

            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0.3, np.pi - 0.3)
            r = np.random.uniform(*radius_range)

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            pose = np.eye(4)
            pose[:3, 3] = [x, y, z]

            poses.append(pose)

        return poses


    # --------------------------------------------------
    # Render mesh
    # --------------------------------------------------

    def render(self, pose):

        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.W, self.H, self.fx, self.fy, self.cx, self.cy
        )

        self.renderer.setup_camera(intrinsics, pose)

        color = np.asarray(self.renderer.render_to_image())
        depth = np.asarray(self.renderer.render_to_depth_image())

        color = cv2.GaussianBlur(color, (3, 3), 0)

        return color, depth


    # --------------------------------------------------
    # Build synthetic dataset
    # --------------------------------------------------

    def build_synthetic_database(self, n_views=200):

        logger.info("Rendering synthetic views")

        poses = self.sample_poses(n_views)

        descriptors = []

        for pose in tqdm(poses, desc="Rendering database"):

            img, depth = self.render(pose)

            small = cv2.resize(img, (80, 60))
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

            desc = gray.flatten().astype(np.float32)

            descriptors.append(desc)
            self.synthetic_views.append((img, depth, pose))

        descriptors = np.stack(descriptors)

        logger.info("Building FAISS index")

        self.faiss_index = faiss.IndexFlatL2(descriptors.shape[1])
        self.faiss_index.add(descriptors)


    # --------------------------------------------------
    # LoFTR matching
    # --------------------------------------------------

    def loftr_match(self, render_img, frame):

        r = cv2.cvtColor(render_img, cv2.COLOR_RGB2GRAY)
        f = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        r = cv2.resize(r, (320, 240))
        f = cv2.resize(f, (320, 240))

        t0 = torch.from_numpy(r / 255.).float()[None, None].to(self.device)
        t1 = torch.from_numpy(f / 255.).float()[None, None].to(self.device)

        with torch.no_grad():
            out = self.loftr({"image0": t0, "image1": t1})

        k0 = out["keypoints0"].cpu().numpy() * 2
        k1 = out["keypoints1"].cpu().numpy() * 2

        return k0, k1


    # --------------------------------------------------
    # Pose estimation
    # --------------------------------------------------

    def estimate_pose(self, k0, k1, depth):

        pts3d = []
        pts2d = []

        for p0, p1 in zip(k0, k1):

            x, y = int(p0[0]), int(p0[1])

            if x < 0 or y < 0 or x >= self.W or y >= self.H:
                continue

            z = depth[y, x]

            if z == 0:
                continue

            X = (x - self.cx) * z / self.fx
            Y = (y - self.cy) * z / self.fy

            pts3d.append([X, Y, z])
            pts2d.append(p1)

        if len(pts3d) < 6:
            return None

        pts3d = np.array(pts3d, np.float32)
        pts2d = np.array(pts2d, np.float32)

        success, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d, pts2d, self.K, None
        )

        if not success:
            return None

        return rvec, tvec


    # --------------------------------------------------
    # Frame processing
    # --------------------------------------------------

    def process_frame(self, frame):

        small = cv2.resize(frame, (80, 60))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        query = gray.flatten().astype(np.float32)

        _, idx = self.faiss_index.search(query.reshape(1, -1), 5)

        best_pose = None
        best_matches = 0

        for i in idx[0]:

            render_img, depth, _ = self.synthetic_views[i]

            k0, k1 = self.loftr_match(render_img, frame)

            matches = len(k0)

            if matches > best_matches:

                pose = self.estimate_pose(k0, k1, depth)

                if pose is None:
                    continue

                best_matches = matches
                best_pose = pose

        if best_pose is not None:
            self.trajectory.append(best_pose)

        return best_pose, best_matches


    # --------------------------------------------------
    # Video tracking
    # --------------------------------------------------

    def track_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        stride = max(int(fps / 5), 1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(range(total_frames))

        frame_id = 0

        for _ in pbar:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % stride != 0:
                frame_id += 1
                continue

            frame = cv2.resize(frame, (self.W, self.H))

            pose, matches = self.process_frame(frame)

            pbar.set_description(
                f"Processing video | LoFTR matches: {matches}"
            )

            frame_id += 1

        cap.release()


# --------------------------------------------------
# Run pipeline
# --------------------------------------------------

tracker = MeshPoseTracker("meshes/data.glb")

tracker.build_synthetic_database(100)

tracker.track_video("data/padu.mp4")

print(tracker.trajectory)