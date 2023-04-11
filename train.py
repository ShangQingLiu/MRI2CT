import os

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from ct_nerf.utils import raw2outputs

from ct_nerf.dataset_new import RayDataset, MultiRayDataset
from ct_nerf.logger import logger
from ct_nerf.network import NeRF
from ct_nerf.parser import config_parser
from ct_nerf.latent_embeddings import init_latent_embeddings


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


    #we now need to load 2 different datasets, a pretrain dataset and a finetune dataset   

    #dataset = MultiRayDataset(args)                                          #data here into dataloader, HERE WE LOAD THE NEW DATASET WITH MULTIPLE VOLUMES
    #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4,)



    pretrain_dataset = MultiRayDataset(args)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4,)


    finetune_dataset = RayDataset(args)     
    finetune_dataloader = DataLoader(finetune_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4,)



    #Embedding INIT
    num_patients = len(pretrain_dataset)
    latent_size = 256

    latent_embeddings = latent_embeddings(num_patients, latent_size)     #INIT HEREEEE

    model = EmbeddingNeRF(
        args.netdepth,
        args.netwidth,
        5,
        multi_res=args.multires,
        latent_dim = latent_dim,
    ).to(device)
    if args.use_wandb:
        wandb.watch(model, log_freq=100)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(list(model.parameters()) + [latent_embeddings], lr=args.lr) #<----- optimize weights AND embedding
    latent_optimizer = torch.optim.Adam([latent_embeddings], lr=args.lr)
    
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

    #pretraining across objects
    for epoch in range(start_epoch, args.N_epoches + 1):
        optimizer.zero_grad()
        for data in pretrain_dataloader:
            B, N, _ = data["pts"].shape
            pts = data["pts"].view(-1, 5).to(device)
            vals = data["vals"].view(-1, 1).to(device)

            # CHANGE: Get the object index and obtain the corresponding latent embeddings
            object_idx = data["object_idx"].to(device)
            object_latent_embedding = latent_embeddings[object_idx]

            # CHANGE: Pass object_latent_embedding to the model
            raw = model(pts, object_latent_embedding)
            loss = args.lambda_vols * crit_mae(raw, vals) * args.std_val / args.N_step_opt

            loss.backward()
            if step % args.N_step_opt == 0:
                optimizer.step()
                optimizer.zero_grad()



    #object specific finetune training
    for epoch in range(start_epoch, args.N_epoches + 1):
        latent_optimizer.zero_grad()
        for data in finetune_dataloader:                             # batch by batch
            B, N, _ = data["pts"].shape                     #batch size b, number of points n
            pts = data["pts"].view(-1, 5).to(device)        #reshaping wiht view()
            vals = data["vals"].view(-1, 1).to(device)      #gt semantic lables?
            inten = data["inten"].view(-1, 1).to(device)    #intensity values


            #modified to incorporate object specific embedding
            object_idx = data["object_idx"].to(device)
            object_latent_embedding = latent_embeddings[object_idx]
            raw = model(pts, object_latent_embedding)



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

            loss.backward()                                 #compute gradients
            if step % args.N_step_opt == 0:                 #only optimize every N_step_opt steps
                latent_optimizer.step()                            #update params
                latent_optimizer.zero_grad()                       #reset gradients to 0

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

            if step % args.N_step_val == 0:                 #eval on validaton set
                with torch.no_grad():
                    maes, preds, gts = [], [], []           #Mean Absolute Error
                    for i in range(args.N_val_planes):
                        pts, vals = dataset.random_plane()  #iterate over random plane from validation dataset
                        H, W, _ = pts.shape
                        pts = pts.reshape(-1, 5).to(device)
                        vals = vals.reshape(-1, 1).to(device)

                        raws = []
                        for i in range(0, H * W, args.chunk_size):    #process data chunkwise for memory fitting 
                            raw = model(pts[i : i + args.chunk_size])
                            raws.append(raw)
                            mae = (
                                crit_mae(raw, vals[i : i + args.chunk_size])
                                * args.std_val
                            )
                            maes.append(mae[None])                      #append processed chunks back together

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
                scheduler.step()                            #update learning rate

            step += 1

        if epoch % args.N_epoch_save == 0:                  #save checkpoints every x epochs
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
