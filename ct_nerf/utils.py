import numpy as np
import torch


def rad2mat(rad):
    return torch.from_numpy(
        np.array(
            [
                [np.cos(rad), np.sin(rad), 0],
                [-np.sin(rad), np.cos(rad), 0],
                [0, 0, 1],
            ]
        )
    ).float()


def get_indexes(H, D, W):
    index_h = torch.linspace(-1.0, 1.0, H + 1)[:-1] + 1 / H
    index_d = torch.linspace(-1.0, 1.0, D + 1)[:-1] + 1 / D
    index_w = torch.linspace(-1.0, 1.0, W + 1)[:-1] + 1 / W
    return [index_h, index_d, index_w]


def get_rays(H, W, theta, rot_mat=None):
    index_h = torch.linspace(-1.0, 1.0, H + 1)[:-1] + 1 / H
    index_d = torch.linspace(-1.0, 1.0, 1)
    index_w = torch.linspace(-1.0, 1.0, W + 1)[:-1] + 1 / W
    grid_h, grid_d, grid_w = torch.meshgrid(
        [index_h, index_d, index_w], indexing="ij"
    )

    rays_o = torch.stack((grid_w, grid_d, grid_h), -1).squeeze()
    rays_d = torch.zeros_like(rays_o)
    rays_d[..., 1] = 1.0

    if rot_mat is None:
        rot_mat = rad2mat(theta / 180 * np.pi)
    rays_o = rays_o.matmul(rot_mat)
    rays_d = rays_d.matmul(rot_mat)

    return rays_o, rays_d


def get_pts(ray_o, ray_d, N_samples, perturb=True):
    z_vals = torch.linspace(0.0, 2.0, N_samples + 1)[:-1]
    if perturb:
        offset = torch.rand(z_vals.shape) * 2 / N_samples
    else:
        offset = 1 / N_samples
    z_vals = z_vals + offset

    pts = ray_o[None, :] + ray_d[None, :] * z_vals[:, None]  # [N, 3]
    return pts


def raw2outputs(raw, raw_noise_std=0):
    """Transforms model's predictions to semantically meaningful values."""
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw.shape).to(raw.device) * raw_noise_std

    inten = raw + noise
    val_map = inten.mean(1)

    return val_map
