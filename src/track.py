import cv2
import torch
import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
import os

from lightglue.utils import load_image, rbd
from kornia.feature import LoFTR
from lightglue import DISK, ALIKED, LightGlue

# Optional RoMa matcher
try:
    from roma import roma_outdoor
    roma_model = roma_outdoor(pretrained=True)
except:
    roma_model = None

os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# Utility
# --------------------------------------------------

def random_rotation_matrix():
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    angle = np.random.uniform(0,2*np.pi)

    K = np.array([
        [0,-axis[2],axis[1]],
        [axis[2],0,-axis[0]],
        [-axis[1],axis[0],0]
    ])

    R = np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)
    return R


# --------------------------------------------------
# Camera parameters
# --------------------------------------------------

W,H = 640,480
fx = fy = 700
cx = W/2
cy = H/2

K = np.array([
    [fx,0,cx],
    [0,fy,cy],
    [0,0,1]
],dtype=np.float32)


# --------------------------------------------------
# Load mesh
# --------------------------------------------------

tm = trimesh.load("meshes/data.glb")

if isinstance(tm,trimesh.Scene):
    tm = trimesh.util.concatenate(tuple(tm.geometry.values()))

vertices = np.asarray(tm.vertices)
faces = np.asarray(tm.faces)


# --------------------------------------------------
# Normalize mesh
# --------------------------------------------------

center = vertices.mean(axis=0)
vertices -= center

scale = max(np.max(np.linalg.norm(vertices,axis=1)),1e-6)
vertices /= scale

mesh = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(vertices),
    triangles=o3d.utility.Vector3iVector(faces)
)


# --------------------------------------------------
# Random mesh rotation
# --------------------------------------------------

R = random_rotation_matrix()

v = np.asarray(mesh.vertices)
v = (R@v.T).T

mesh.vertices = o3d.utility.Vector3dVector(v)
mesh.compute_vertex_normals()


# --------------------------------------------------
# Renderer
# --------------------------------------------------

renderer = o3d.visualization.rendering.OffscreenRenderer(W,H)

material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"
material.base_color = [1,1,1,1]

renderer.scene.add_geometry("mesh",mesh,material)

intrinsics = o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)

extrinsic = np.eye(4)
extrinsic[2,3] = 2.2

renderer.setup_camera(intrinsics,extrinsic)

cam_pos = extrinsic[:3,3]
light_dir = -cam_pos/np.linalg.norm(cam_pos)

renderer.scene.scene.set_sun_light(
    direction=light_dir.tolist(),
    color=[1,1,1],
    intensity=250000
)

renderer.scene.scene.enable_sun_light(True)
renderer.scene.scene.set_indirect_light_intensity(60000)
renderer.scene.set_background([0.9,0.9,0.9,1])


# --------------------------------------------------
# Render mesh
# --------------------------------------------------

color_image = renderer.render_to_image()
depth_image = renderer.render_to_depth_image()

render_img = np.asarray(color_image)
depth = np.asarray(depth_image)

render_img = cv2.GaussianBlur(render_img,(3,3),0)

cv2.imwrite("outputs/render.png",render_img)
np.save("outputs/depth.npy",depth)


# --------------------------------------------------
# Load real image
# --------------------------------------------------

real_img = load_image("image.png")
real_np = real_img.permute(1,2,0).cpu().numpy()

real_vis = (real_np*255).astype(np.uint8)
real_vis = cv2.resize(real_vis,(W,H))


# --------------------------------------------------
# LoFTR Matching
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

matcher_loftr = LoFTR(pretrained="outdoor").to(device).eval()

render_gray = cv2.cvtColor(render_img,cv2.COLOR_RGB2GRAY)
real_gray = cv2.cvtColor(real_vis,cv2.COLOR_RGB2GRAY)

render_gray = torch.from_numpy(render_gray/255.).float()[None,None].to(device)
real_gray = torch.from_numpy(real_gray/255.).float()[None,None].to(device)

with torch.no_grad():
    out = matcher_loftr({"image0":render_gray,"image1":real_gray})

kpts0_loftr = out["keypoints0"].cpu().numpy()
kpts1_loftr = out["keypoints1"].cpu().numpy()

print("LoFTR matches:",len(kpts0_loftr))


# --------------------------------------------------
# DISK + ALIKED features
# --------------------------------------------------

disk = DISK(max_num_keypoints=2048).eval().to(device)
aliked = ALIKED(max_num_keypoints=2048).eval().to(device)

matcher_disk = LightGlue(features="disk").eval().to(device)
matcher_aliked = LightGlue(features="aliked").eval().to(device)

render_tensor = torch.from_numpy(render_img).float()/255.
render_tensor = render_tensor.permute(2,0,1).to(device)

real_tensor = torch.from_numpy(real_vis).float()/255.
real_tensor = real_tensor.permute(2,0,1).to(device)

with torch.no_grad():

    fr_d = rbd(disk.extract(render_tensor))
    fi_d = rbd(disk.extract(real_tensor))

    fr_a = rbd(aliked.extract(render_tensor))
    fi_a = rbd(aliked.extract(real_tensor))


with torch.no_grad():

    md = rbd(matcher_disk({
        "image0":{"keypoints":fr_d["keypoints"].unsqueeze(0),
                  "descriptors":fr_d["descriptors"].unsqueeze(0)},
        "image1":{"keypoints":fi_d["keypoints"].unsqueeze(0),
                  "descriptors":fi_d["descriptors"].unsqueeze(0)}
    }))

    ma = rbd(matcher_aliked({
        "image0":{"keypoints":fr_a["keypoints"].unsqueeze(0),
                  "descriptors":fr_a["descriptors"].unsqueeze(0)},
        "image1":{"keypoints":fi_a["keypoints"].unsqueeze(0),
                  "descriptors":fi_a["descriptors"].unsqueeze(0)}
    }))


k0_d = fr_d["keypoints"].cpu().numpy()
k1_d = fi_d["keypoints"].cpu().numpy()
idx_d = md["matches"].cpu().numpy()

k0_a = fr_a["keypoints"].cpu().numpy()
k1_a = fi_a["keypoints"].cpu().numpy()
idx_a = ma["matches"].cpu().numpy()

disk0 = k0_d[idx_d[:,0]]
disk1 = k1_d[idx_d[:,1]]

aliked0 = k0_a[idx_a[:,0]]
aliked1 = k1_a[idx_a[:,1]]


# --------------------------------------------------
# Merge all matches
# --------------------------------------------------

kpts0 = np.vstack([kpts0_loftr,disk0,aliked0])
kpts1 = np.vstack([kpts1_loftr,disk1,aliked1])

print("Total matches:",len(kpts0))


# --------------------------------------------------
# Draw matches
# --------------------------------------------------

canvas = np.hstack([render_img,real_vis])
offset = render_img.shape[1]

for p0,p1 in zip(kpts0,kpts1):

    p0 = tuple(p0.astype(int))
    p1 = tuple((p1 + np.array([offset,0])).astype(int))

    color = tuple(np.random.randint(0,255,3).tolist())
    cv2.line(canvas,p0,p1,color,1)

cv2.imwrite("outputs/matches.png",canvas)


# --------------------------------------------------
# Build 3D-2D correspondences
# --------------------------------------------------

pts3d=[]
pts2d=[]

for p0,p1 in zip(kpts0,kpts1):

    x,y = int(p0[0]),int(p0[1])

    if x<0 or x>=depth.shape[1] or y<0 or y>=depth.shape[0]:
        continue

    if depth[y,x]==0:
        continue

    Z = depth[y,x]
    X = (x-cx)*Z/fx
    Y = (y-cy)*Z/fy

    pts3d.append([X,Y,Z])
    pts2d.append(p1)

pts3d = np.array(pts3d,dtype=np.float32)
pts2d = np.array(pts2d,dtype=np.float32)

print("3D correspondences:",len(pts3d))


# --------------------------------------------------
# Pose estimation
# --------------------------------------------------

if len(pts3d) > 6:

    success,rvec,tvec,inliers = cv2.solvePnPRansac(
        pts3d,pts2d,K,None,
        reprojectionError=6,
        confidence=0.99,
        iterationsCount=1000
    )

    if inliers is not None:
        print("PnP inliers:",len(inliers))

    print("Pose estimated:",success)
    print("Translation:",tvec.flatten())

    proj,_ = cv2.projectPoints(
        pts3d.reshape(-1,1,3),
        rvec,tvec,K,None
    )

    proj = proj.reshape(-1,2)

    img = real_vis.copy()

    for p in proj:
        x,y = int(p[0]),int(p[1])
        if 0<=x<img.shape[1] and 0<=y<img.shape[0]:
            cv2.circle(img,(x,y),2,(0,255,0),-1)

    cv2.imwrite("outputs/reprojection.png",img)

    plt.imshow(img)
    plt.title("Reprojected 3D points")
    plt.axis("off")
    plt.show()