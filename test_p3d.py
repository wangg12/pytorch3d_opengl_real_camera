# test pytorch 3d renderer
# render multi objects in batch, one in one image
import errno
import os
import os.path as osp
import sys
import time
import struct
import mmcv
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2quat, mat2euler, quat2euler
from transforms3d.quaternions import axangle2quat, mat2quat, qinverse, qmult

# io utils
from pytorch3d.io import load_objs_as_meshes, load_obj

# rendering components
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    OpenGLPerspectiveCameras,
    HardPhongShader,
    PointLights,
    RasterizationSettings,
    SoftSilhouetteShader,
    look_at_rotation,
    look_at_view_transform,
)
from pytorch3d.renderer.cameras import SfMPerspectiveCameras
from cameras_real import OpenGLRealPerspectiveCameras

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate


def grid_show(ims, titles=None, row=1, col=3, dpi=200, save_path=None, title_fontsize=5, show=True):
    if row * col < len(ims):
        print("_____________row*col < len(ims)___________")
        col = int(np.ceil(len(ims) / row))
    fig = plt.figure(dpi=dpi, figsize=plt.figaspect(row / float(col)))
    k = 0
    for i in range(row):
        for j in range(col):
            plt.subplot(row, col, k + 1)
            plt.axis("off")
            plt.imshow(ims[k])
            if titles is not None:
                # plt.title(titles[k], size=title_fontsize)
                plt.text(
                    0.5,
                    1.08,
                    titles[k],
                    horizontalalignment="center",
                    fontsize=title_fontsize,
                    transform=plt.gca().transAxes,
                )
            k += 1
            if k == len(ims):
                break
    # plt.tight_layout()
    if show:
        plt.show()
    else:
        if save_path is not None:
            mmcv.mkdir_or_exist(osp.dirname(save_path))
            plt.savefig(save_path)
    return fig


data_root = "Render/OBJ"
K = np.array(
    [
        [496.0020141601562500, 0, 238.6889953613281250],
        [0, 496.0020141601562500, 322.0400085449218750],
        [0, 0, 1],
    ],
    dtype=np.float32,
)

pose = np.loadtxt(osp.join(data_root, "00000_pose.txt"))
R = pose[:3, :3]
t = pose[:3, 3]

W = 480
H = 640
ZNEAR = 0.1
ZFAR = 10.0
IMG_SIZE = max(W, H)

obj_paths = [osp.join(data_root, "194725.obj")]
texture_paths = [osp.join(data_root, "194725.png")]


def print_stat(data, name=""):
    print(
        name,
        "min",
        data.min(),
        "max",
        data.max(),
        "mean",
        data.mean(),
        "std",
        data.std(),
        "sum",
        data.sum(),
        "shape",
        data.shape,
    )


###################################################################################################


def main():
    # Set the cuda device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    mesh = load_objs_as_meshes(obj_paths, device=device)
    texture_image = mesh.textures.maps_padded()

    cameras = OpenGLRealPerspectiveCameras(
        focal_length=((K[0, 0], K[1, 1]),),  # Nx2
        principal_point=((K[0, 2], K[1, 2]),),  # Nx2
        x0=0,
        y0=0,
        w=H,
        h=H,  # HEIGHT,
        znear=ZNEAR,
        zfar=ZFAR,
        device=device,
    )

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 640x640. To form the blended image we use 100 faces for each pixel. Refer to rasterize_meshes.py
    # for an explanation of this parameter.
    silhouette_raster_settings = RasterizationSettings(
        image_size=IMG_SIZE,  # longer side or scaled longer side
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=100,  # the nearest faces_per_pixel points along the z-axis.
        bin_size=0,
    )
    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=silhouette_raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )
    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    phong_raster_settings = RasterizationSettings(
        image_size=IMG_SIZE, blur_radius=0.0, faces_per_pixel=1, bin_size=0
    )
    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=phong_raster_settings),
        shader=HardPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    batch_R = torch.tensor(np.stack([R]), device=device, dtype=torch.float32).permute(
        0, 2, 1
    )  # Bx3x3
    batch_T = torch.tensor(np.stack([t]), device=device, dtype=torch.float32)  # Bx3

    silhouete = silhouette_renderer(meshes_world=mesh, R=batch_R, T=batch_T)
    image_ref = phong_renderer(meshes_world=mesh, R=batch_R, T=batch_T)
    # crop results
    silhouete = silhouete[:, :H, :W, :].cpu().numpy()
    image_ref = image_ref[:, :H, :W, :3].cpu().numpy()

    pred_images = image_ref

    opengl = mmcv.imread(osp.join("Render/OpenGL.png"), "color") / 255.0

    for i in range(pred_images.shape[0]):
        pred_mask = silhouete[i, :, :, 3].astype("float32")

        print("num rendered images", pred_images.shape[0])
        image = pred_images[i]

        diff_opengl = np.abs(opengl[:, :, ::-1].astype("float32") - image.astype("float32"))
        print("image", image.shape, image.min(), image.max())

        print("dr mask area: ", pred_mask.sum())

        print_stat(pred_mask, "pred_mask")
        show_ims = [image, diff_opengl, opengl[:, :, ::-1]]
        show_titles = ["image", "diff_opengl", "opengl"]
        grid_show(show_ims, show_titles, row=1, col=3)


if __name__ == "__main__":
    main()
