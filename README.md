Below is the **same pipeline rewritten to use Open3D instead of pyrender** for mesh rendering. The rest of the workflow (ALIKED → LightGlue → depth → PnP) stays the same.

---

# Cell 1 — Install dependencies

```python
!pip install torch lightglue opencv-python trimesh open3d
```

---

# Cell 2 — Imports

```python
import cv2
import torch
import numpy as np
import trimesh
import open3d as o3d

from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd
```

---

# Cell 3 — Camera parameters

```python
W, H = 640, 480

fx = fy = 700
cx = W / 2
cy = H / 2

K = np.array([
    [fx,0,cx],
    [0,fy,cy],
    [0,0,1]
])
```

---

# Cell 4 — Load mesh

```python
mesh_trimesh = trimesh.load("mesh.obj")

print(mesh_trimesh.vertices.shape)
print(mesh_trimesh.faces.shape)
```

Convert to Open3D mesh:

```python
mesh_o3d = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(mesh_trimesh.vertices),
    triangles=o3d.utility.Vector3iVector(mesh_trimesh.faces)
)

mesh_o3d.compute_vertex_normals()
```

---

# Cell 5 — Render mesh with Open3D

```python
renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)

material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"

renderer.scene.add_geometry("mesh", mesh_o3d, material)
```

Set camera:

```python
intrinsics = o3d.camera.PinholeCameraIntrinsic(
    W, H, fx, fy, cx, cy
)

extrinsic = np.eye(4)
extrinsic[2,3] = 2.5  # move camera back

renderer.setup_camera(intrinsics, extrinsic)
```

Render:

```python
color = renderer.render_to_image()
depth_img = renderer.render_to_depth_image()

render_img = np.asarray(color)
depth = np.asarray(depth_img)

cv2.imwrite("render.png", render_img)
```

---

# Cell 6 — Load real image

```python
real_img = load_image("image.png")
```

---

# Cell 7 — Feature extraction

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = ALIKED(max_num_keypoints=1024).eval().to(device)

render_tensor = load_image("render.png").to(device)
real_tensor = real_img.to(device)

feats_render = rbd(extractor.extract(render_tensor))
feats_real = rbd(extractor.extract(real_tensor))

print("render keypoints:", feats_render["keypoints"].shape)
print("real keypoints:", feats_real["keypoints"].shape)
```

---

# Cell 8 — LightGlue matching

```python
matcher = LightGlue(features="aliked").eval().to(device)

matches = matcher({
    "image0": {
        "keypoints": feats_render["keypoints"][None],
        "descriptors": feats_render["descriptors"][None],
    },
    "image1": {
        "keypoints": feats_real["keypoints"][None],
        "descriptors": feats_real["descriptors"][None],
    }
})

matches = rbd(matches)

idx0 = matches["matches"][:,0]
idx1 = matches["matches"][:,1]

print("matches:", len(idx0))
```

---

# Cell 9 — Convert render pixels → 3D points

```python
render_kpts = feats_render["keypoints"][idx0].cpu().numpy()
real_kpts = feats_real["keypoints"][idx1].cpu().numpy()

points3D = []
points2D = []

for p_render, p_real in zip(render_kpts, real_kpts):

    x = int(p_render[0])
    y = int(p_render[1])

    if x < 0 or x >= W or y < 0 or y >= H:
        continue

    z = depth[y, x]

    if z == 0:
        continue

    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z

    points3D.append([X, Y, Z])
    points2D.append(p_real)

points3D = np.array(points3D, dtype=np.float32)
points2D = np.array(points2D, dtype=np.float32)

print("3D points:", points3D.shape)
```

---

# Cell 10 — Solve pose with PnP

```python
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    points3D,
    points2D,
    K,
    None
)

print("success:", success)
print("inliers:", len(inliers))

print("rvec:", rvec)
print("tvec:", tvec)
```

---

# Cell 11 — Visualize reprojection

```python
img = cv2.imread("image.png")

proj,_ = cv2.projectPoints(
    points3D,
    rvec,
    tvec,
    K,
    None
)

proj = proj.reshape(-1,2)

for p in proj:
    x,y = int(p[0]),int(p[1])

    if 0 <= x < W and 0 <= y < H:
        cv2.circle(img,(x,y),2,(0,255,0),-1)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis("off")
```

---

# Key improvement vs the earlier code

The rendering stage is now:

```
mesh → Open3D renderer → RGB + depth
```

instead of:

```
mesh → pyrender → RGB + depth
```

Open3D tends to be **more stable locally**, especially on CPU-only systems.

---

If you want, the **next improvement** would make this pipeline **much more accurate** by adding **pose refinement (SE(3) optimization)** after the initial PnP estimate.
