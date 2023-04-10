import numpy as np
from PIL import Image

from ct_nerf.dataset import RayDataset
from ct_nerf.parser import config_parser


def get_dataset():
    parser = config_parser()
    args = parser.parse_args()
    args.perturb = True
    args.N_samples = 512

    return RayDataset(args)


def test_training():
    dataset = get_dataset()
    data = dataset[36452]
    for key, val in data.items():
        print(key, val.shape)
    for i in np.random.choice(128 * 512 * 180, 1):
        data = dataset[i]
        print(i, data['vals'].mean(), data['inten'])


def test_validation():
    dataset = get_dataset()
    pts, vals = dataset.random_plane()
    print(pts)
    print(vals)
    print(pts.shape)
    print(vals.shape)
    vals = vals.numpy()
    vals = Image.fromarray((vals * 255).astype(np.uint8))
    vals.save("test.png")


def test_testing():
    dataset = get_dataset()
    pts, vals = dataset.all_planes()
    print(pts.shape)
    print(vals.shape)


if __name__ == "__main__":
    test_training()
    # test_validation()
    # test_testing()
