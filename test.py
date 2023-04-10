import os

import SimpleITK as sitk
import torch

from ct_nerf.dataset import RayDataset
from ct_nerf.logger import logger
from ct_nerf.network import NeRF
from ct_nerf.parser import config_parser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test():
    parser = config_parser()
    args = parser.parse_args()

    dataset = RayDataset(args)
    model = NeRF(
        args.netdepth,
        args.netwidth,
        5,
        multi_res=args.multires,
    ).to(device)

    crit_mae = torch.nn.L1Loss().to(device)
    crit_mse = torch.nn.MSELoss().to(device)

    assert args.ckpt_path is not None
    ckpts = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpts["model"])

    with torch.no_grad():
        pts, vals = dataset.all_planes()
        H, D, W, _ = pts.shape
        pts = pts.view(-1, 5).to(device)
        vals = vals.view(-1, 1).to(device)
        maes, planes = [], []

        for i in range(0, H * D * W, args.chunk_size):
            raw = model(pts[i : i + args.chunk_size])
            mae = crit_mae(raw, vals[i : i + args.chunk_size]) * args.std_val
            maes.append(mae.view(-1))
            logger.info(
                "==Test== MAE-{}: {:4f}".format(
                    i // args.chunk_size, mae.item()
                )
            )

            plane = raw.to("cpu")
            planes.append(plane)

        mae = torch.cat(maes).mean()
        logger.info("==Test== MAE-ave: {:.4f}".format(mae.item()))

        tensor = torch.cat(planes, 0).view(H, D, W)
        vol = dataset.tensor2vol(tensor)
        folder = os.path.split(args.ckpt_path)[0]
        path = os.path.join(folder, "recon.mha")
        sitk.WriteImage(vol, path)


if __name__ == "__main__":
    test()
