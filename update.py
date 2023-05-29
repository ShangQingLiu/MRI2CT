import os
import glob

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  

from sklearn.decomposition import PCA


from ct_nerf.utils import raw2outputs

from ct_nerf.dataset_new import RayDataset, MultiRayDataset
from ct_nerf.logger import logger
from ct_nerf.network import EmbeddingNeRF, NeRF
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

    #TODO: Dataset & Dataloader
    pretrain_dataset = MultiRayDataset(args)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4,)

    #Embedding INIT
    print(f"rays.shape:{pretrain_dataset.rays_o.shape}")
    #print(f"pretrain_dataset.volume_indices:{pretrain_dataset.total_volumes}")

    if args.mode == 0:
        latent_size = args.latent_size
    elif args.mode == 1:
        latent_size = args.netwidth
        
    matrix_size = args.matrix_size
    latent_embeddings = init_latent_embeddings(pretrain_dataset.total_volumes, latent_size, matrix_size, device=device)    

    model = EmbeddingNeRF(
        D=args.netdepth,  # This matches D in the model definition
        W=args.netwidth,  # This matches W in the model definition
        input_ch=5,  # This matches input_ch in the model definition
        latent_dim=latent_size,  # This matches latent_dim in the model definition
        output_ch=1,  # This matches output_ch in the model definition
        skips=[4],  # This matches skips in the model definition
        multi_res=args.multires,  # This matches multi_res in the model definition
        matrix_size=args.matrix_size,  # This matches matrix_size in the model definition
        mode=args.mode  # This matches mode in the model definition
    ).to(device)

    
    if args.use_wandb:
        wandb.watch(model, log_freq=100)

    #optimizer = torch.optim.Adam(list(model.parameters()) + list(latent_embeddings.parameters()), lr=args.lr) #<----- optimize weights AND embedding
    latent_optimizer = torch.optim.Adam(list(latent_embeddings.parameters()), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        # optimizer, gamma=0.1 ** (1 / args.N_epoches / args.lr_decay_ratio)
        latent_optimizer,
        gamma=0.1 ** (1 / 300),
    )

    crit_mae = torch.nn.L1Loss().to(device)
    crit_mse = torch.nn.MSELoss().to(device)
    start_epoch = 0


    #LOAD CHECKPOINT, Find the checkpoint with the same run name

    logs_dir = os.path.join(basedir)
    all_subdirs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]

    matching_subdir = None
    for subdir in all_subdirs:
        if args.name == subdir:
            matching_subdir = subdir
            break

    print("RESUME?", args.resume)
    if matching_subdir is not None:
        # If a matching subdirectory is found, search for the latest .pth file within it
        matching_logs_dir = os.path.join(logs_dir, matching_subdir)
        all_checkpoints = glob.glob(os.path.join(matching_logs_dir, "*.pth"))
        if len(all_checkpoints) > 0:
            latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
            print(f"Loading checkpoint from {latest_checkpoint}")
            ckpts = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(ckpts["model"])
            #latent_embeddings.load_state_dict(ckpts["latent_embeddings"]) #load embeddings if saved
        
            if args.resume:
                latent_optimizer.load_state_dict(ckpts["opt"])
                scheduler.load_state_dict(ckpts["sche"])
                start_epoch = ckpts["epoch"]
                print("Starting from epoch", start_epoch)

    elif args.ckpt_path is not None:      #if specifc checkpoint specified:        logs/test_lat_3vol_512/1.pth
        print(f"Loading checkpoint from {args.ckpt_path}")
        ckpts = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpts["model"])
    
    else:
        print("No matching checkpoints found.")
   

    start_epoch += 1
    step = 1

    print("Number of steps per epoch:", len(pretrain_dataset) / args.batch_size)


    tb_writer = SummaryWriter(os.path.join(basedir, name, 'logs'))

    #pretraining across objects
    for epoch in range(start_epoch, args.N_epoches + 1):
        

        latent_optimizer.zero_grad()
        for data in pretrain_dataloader:
            B, N, _ = data["pts"].shape
            pts = data["pts"].view(-1, 5).to(device)
            vals = data["vals"].view(-1, 1).to(device)
            inten = data["inten"].view(-1, 1).to(device)

            object_idx = data["object_idx"].to(device)    

            object_latent_embedding = latent_embeddings(object_idx.to(torch.int64).to(device))
            object_latent_embedding = object_latent_embedding.squeeze()
            object_latent_embedding = object_latent_embedding.repeat_interleave(N, dim=0)
            
            raw = model(pts, object_latent_embedding)

            output = raw2outputs(raw.view(B, N), args.raw_noise_std / args.std_val)[..., None]
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
                latent_optimizer.step()
                latent_optimizer.zero_grad()
                


            if step % args.N_step_log == 0:

                # Compute the average L2-norm of the latent embeddings
                avg_l2_norm = torch.norm(latent_embeddings.weight, dim=1).mean().item()

                metrics = {
                    "lr": latent_optimizer.param_groups[0]["lr"],
                    "MSE": loss_mse.item(),
                    "MAE": loss_mae.item(),
                    "Proj": loss_proj.item(),
                    "Latent_Avg_L2_Norm": avg_l2_norm
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

                for key, val in metrics.items():
                    tb_writer.add_scalar('train/' + key, val, step)
                if args.use_wandb:
                    wandb.log(metrics, step)
                
                # Visualize latent embeddings using PCA
                latent_np = latent_embeddings.weight.detach().cpu().numpy()

                if latent_np.shape[0] > 1:
                    pca = PCA(n_components=2)
                    latent_pca = pca.fit_transform(latent_np)

                    for i, latent_emb in enumerate(latent_pca):
                        tb_writer.add_scalar(f'LatentEmbedding/PC1_{i}', latent_emb[0], step)
                        tb_writer.add_scalar(f'LatentEmbedding/PC2_{i}', latent_emb[1], step)
                else:
                    tb_writer.add_scalar(f'LatentEmbedding/PC1_0', latent_np[0, 0], step)
                    tb_writer.add_scalar(f'LatentEmbedding/PC2_0', latent_np[0, 1], step)


            if step % args.N_step_val == 0:
                with torch.no_grad():
                    maes, preds, gts = [], [], []
                    for i in range(args.N_val_planes):
                        pts, vals = pretrain_dataset.random_plane()
                        H, W, _ = pts.shape
                        pts = pts.reshape(-1, 5).to(device)
                        vals = vals.reshape(-1, 1).to(device)

                        raws = []


                        object_idx = data["object_idx"].to(device)  
                        object_latent_embedding = latent_embeddings(object_idx.to(torch.int64).to(device))
                        object_latent_embedding = object_latent_embedding.squeeze()
                        object_latent_embedding = object_latent_embedding.repeat_interleave(N, dim=0)

                        #print("OBBBBBBB", object_idx.shape)
                        for i in range(0, H * W, args.chunk_size):
                            raw = model(pts[i : i + args.chunk_size], object_latent_embedding)

                            
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
                    "opt": latent_optimizer.state_dict(),
                    "sche": scheduler.state_dict(),
                    #"latent_embeddings": latent_embeddings.state_dict(), #if they need to be saved?
                },
                path,
            )
            logger.info("Saved checkpoints at {}".format(path))
    tb_writer.close()


if __name__ == "__main__":
    train()

