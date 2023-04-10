import os

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from ct_nerf.utils import raw2outputs

from ct_nerf.dataset import RayDataset
from ct_nerf.logger import logger
from ct_nerf.network import NeRF
from ct_nerf.parser import config_parser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    parser = config_parser()
    args = parser.parse_args()
    if args.use_wandb:
        import wandb

        wandb.init(project="INR", name=args.name)

    basedir = args.basedir
    name = args.name
    if not os.path.isdir(os.path.join(basedir, name)):
        os.makedirs(os.path.join(basedir, name))

    dataset = RayDataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    model = NeRF(
        args.netdepth,
        args.netwidth,
        5,
        multi_res=args.multires,
    ).to(device)
    if args.use_wandb:
        wandb.watch(model, log_freq=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        # optimizer, gamma=0.1 ** (1 / args.N_epoches / args.lr_decay_ratio)
        optimizer,
        gamma=0.1 ** (1 / 300),
    )

    crit_mae = torch.nn.L1Loss().to(device)
    crit_mse = torch.nn.MSELoss().to(device)
    start_epoch = 0

    if args.ckpt_path is not None:
        ckpts = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpts["model"])
        if args.resume:
            optimizer.load_state_dict(ckpts["opt"])
            scheduler.load_state_dict(ckpts["sche"])
            start_epoch = ckpts["epoch"]

    start_epoch += 1
    step = 1
    for epoch in range(start_epoch, args.N_epoches + 1):
        optimizer.zero_grad()
        for data in dataloader:
            B, N, _ = data["pts"].shape
            pts = data["pts"].view(-1, 5).to(device)
            vals = data["vals"].view(-1, 1).to(device)
            inten = data["inten"].view(-1, 1).to(device)

            raw = model(pts)
            output = raw2outputs(
                raw.view(B, N), args.raw_noise_std / args.std_val
            )[..., None]

            loss_mse = (
                args.lambda_vols * crit_mse(raw, vals) * args.std_val**2
            )
            loss_mae = args.lambda_vols * crit_mae(raw, vals) * args.std_val
            loss_proj = (
                args.lambda_proj * crit_mae(output, inten) * args.std_val
            )
            loss = (loss_mae if args.use_grid else loss_proj) / args.N_step_opt

            loss.backward()
            if step % args.N_step_opt == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % args.N_step_log == 0:
                metrics = {
                    "lr": optimizer.param_groups[0]["lr"],
                    "MSE": loss_mse.item(),
                    "MAE": loss_mae.item(),
                    "Proj": loss_proj.item(),
                }
                logger.info(
                    "==Train== epoch: {}, step: {}, ".format(epoch, step)
                    + ", ".join(
                        [
                            "{}: {:4f}".format(key, val)
                            for key, val in metrics.items()
                        ]
                    ),
                )
                if args.use_wandb:
                    wandb.log(metrics, step)

            if step % args.N_step_val == 0:
                with torch.no_grad():
                    maes, preds, gts = [], [], []
                    for i in range(args.N_val_planes):
                        pts, vals = dataset.random_plane()
                        H, W, _ = pts.shape
                        pts = pts.reshape(-1, 5).to(device)
                        vals = vals.reshape(-1, 1).to(device)

                        raws = []
                        for i in range(0, H * W, args.chunk_size):
                            raw = model(pts[i : i + args.chunk_size])
                            raws.append(raw)
                            mae = (
                                crit_mae(raw, vals[i : i + args.chunk_size])
                                * args.std_val
                            )
                            maes.append(mae[None])

                        pred = torch.cat(raws)
                        preds.append(pred.view(H, W))
                        gts.append(vals.view(H, W))

                    mae = torch.cat(maes).mean()
                    logger.info("==Test== MAE-ave: {:4f}".format(mae.item()))

                    if args.use_wandb:
                        pred = torch.cat(preds, 1)
                        gt = torch.cat(gts, 1)
                        visuals = torch.cat((pred, gt), 0)
                        wandb.log({"plane": wandb.Image(visuals * 1020)}, step)
                        logger.info("Draw images at step: {}".format(step))

            if step % 100 == 0:
                scheduler.step()

            step += 1

        if epoch % args.N_epoch_save == 0:
            path = os.path.join(basedir, name, "{}.pth".format(epoch))
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "sche": scheduler.state_dict(),
                },
                path,
            )
            logger.info("Saved checkpoints at {}".format(path))


if __name__ == "__main__":
    train()
