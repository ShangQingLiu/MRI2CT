import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--config", is_config_file=True, help="config file path"
    )
    parser.add_argument("--name", type=str, help="experiment name")
    parser.add_argument(
        "-- /dir",
        type=str,
        default="./logs/",
        help="where to store ckpts and logs",
    )
    parser.add_argument(
        "--vol_path",
        type=str,
        default="/cluster/project/jbuhmann/xiali/datasets/4DCT/07-02-2003-NA-p4-14571/ph10.mha",
        help="input data path",
    )
    parser.add_argument(
        "--proj_dir",
        type=str,
        default="/cluster/project/jbuhmann/xiali/datasets/4DCT/07-02-2003-NA-p4-14571/ph10-projs",
        help="input data path",
    )
    parser.add_argument(
        "--min_val",
        type=int,
        default=-1000,
        help="min value for the original CT",
    )
    parser.add_argument(
        "--std_val",
        type=int,
        default=4096,
        help="range value for the original CT",
    )
    parser.add_argument(
        "--angles",
        type=int,
        default=180,
        help="number of angles to be sampled",
    )

    # training options
    parser.add_argument(
        "--netdepth", type=int, default=8, help="layers in network"
    )
    parser.add_argument(
        "--netwidth", type=int, default=256, help="channels per layer"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="batch size",
    )
    parser.add_argument(
        "--N_epoches",
        type=int,
        default=2,
        help="epoch num in total",
    )
    parser.add_argument(
        "--use_grid",
        action="store_true",
        help="whether to use grid or projection for training",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="whether to use wandb",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to resume from the stored epoch",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--lr_decay_ratio",
        type=float,
        default=1.25,
        help="exponential learning rate ratio",
    )
    parser.add_argument(
        "--lambda_vols",
        type=float,
        default=1.0,
        help="weight for the MAE loss",
    )
    parser.add_argument(
        "--lambda_proj",
        type=float,
        default=1.0,
        help="weight for the MSE loss",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512 * 64,
        help=(
            "number of pts sent through network in parallel, decrease if"
            " running out of memory"
        ),
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )

    # rendering options
    parser.add_argument(
        "--N_samples",
        type=int,
        default=64,
        help="number of coarse samples per ray",
    )
    parser.add_argument(
        "--perturb",
        type=bool,
        default=True,
        help="set to False for no jitter, True for jitter",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=1.0,
        help=(
            "std dev of noise added to regularize sigma_a output, 1e0"
            " recommended"
        ),
    )

    # logging/saving options
    parser.add_argument(
        "--N_step_log",
        type=int,
        default=10,
        help="frequency of optimization over steps",
    )
    parser.add_argument(
        "--N_step_opt",
        type=int,
        default=1,
        help="frequency of optimization over steps",
    )
    parser.add_argument(
        "--N_step_val",
        type=int,
        default=400,
        help="frequency of validation per steps",
    )
    parser.add_argument(
        "--N_val_planes",
        type=int,
        default=4,
        help="number of validation images",
    )
    parser.add_argument(
        "--N_epoch_save",
        type=int,
        default=1,
        help="frequency of weight ckpt saving over epoches",
    )

    return parser
