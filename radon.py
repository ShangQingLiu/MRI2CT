import os

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from PIL import Image

from ct_nerf.utils import get_indexes, rad2mat


def get_vol(vol_path):
    vol = sitk.ReadImage(vol_path)
    vol = sitk.GetArrayFromImage(vol)
    vol = torch.from_numpy(vol).float()
    vol = (vol + 1000) / 4096
    return vol


def get_proj(vol, angle):
    H, D, W = vol.shape
    index_h, index_d, index_w = get_indexes(H, D, W)
    grid_h, grid_d, grid_w = torch.meshgrid(
        [index_h, index_d, index_w], indexing="ij"
    )
    pts = torch.stack((grid_w, grid_d, grid_h), -1)
    rot_mat = rad2mat(angle / 180 * np.pi)
    pts = pts.matmul(rot_mat)
    vals = F.grid_sample(
        vol[None, None], pts[None], align_corners=False
    ).squeeze_()
    proj = vals.mean(dim=1)
    return proj


def main(vol_path, save_dir):
    vol = get_vol(vol_path)
    for angle in range(180):
        proj = get_proj(vol, angle).numpy()

        save_name = "{:03d}.npy".format(angle)
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, proj)

        save_name = "{:03d}.png".format(angle)
        save_path = os.path.join(save_dir, save_name)
        img = Image.fromarray((proj * 255).astype(np.uint8))
        img.save(save_path)
        print(angle, save_name)


if __name__ == '__main__':
    vol_path = '../../datasets/4DCT/07-02-2003-NA-p4-14571/ph10.mha'
    save_dir = '../../datasets/4DCT/07-02-2003-NA-p4-14571/ph10-projs'
    main(vol_path, save_dir)
