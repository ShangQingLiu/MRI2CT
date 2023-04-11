import numpy as np
import numpy.random as random
import os
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import get_indexes, get_pts, get_rays, rad2mat


class MultiRayDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N_samples = args.N_samples                             #nr of samples along each ray
        self.perturb = args.perturb                                 #ammount of perturbation to ray direction vectors?
        self.valid_views = ["axial", "sagital", "coronal"]
        self.mode = mode


        # CHANGE: Added self.volume_indices
        self.volume_indices = []


        
        ###my addition

         # Check if a list of volume paths is provided
        if isinstance(args.volumes, list):
            volume_paths = args.volumes
        # Else, get the list of volumes in the directory
        else:
            volume_paths = [os.path.join(args.volumes_dir, vol) for vol in os.listdir(args.volumes_dir)]

        self.volumes = []               #contains all 3d volumes loaded
        self.props = []                 #list of properties for each volume

        for vol_path in volume_paths:

            vol = sitk.ReadImage(args.vol_path)                         #3d image
            self.props = [
                vol.GetSize(),
                vol.GetOrigin(),
                vol.GetSpacing(),
                vol.GetDirection(),
            ]
            self.props.append(props)

            vol = sitk.GetArrayFromImage(vol)
            self.vol = (torch.from_numpy(vol) - args.min_val) / args.std_val
            self.volumes.append(vol)


            # DONT THINK WE NEED THIS CHANGE: Append the current object index for each ray
            # self.volume_indices += [len(self.volumes) - 1] * len(self.rays_o)

        self.H, self.D, self.W = self.vol.shape
        self.img_size = self.H * self.W

        self.thetas = []
        self.matrixs = {}
        rays_o_list = []
        rays_d_list = []
        intens_list = []

        for ang_ind in range(args.angles):
            theta = int(ang_ind / args.angles * 180)
            self.thetas.append(theta)
            mat = rad2mat(theta / 180 * np.pi)
            self.matrixs[theta] = mat

            rays_o, rays_d = get_rays(self.H, self.W, theta, mat)
            rays_o_list.append(rays_o)
            rays_d_list.append(rays_d)

            proj_name = "{:03d}.npy".format(theta)
            proj_path = os.path.join(args.proj_dir, proj_name)
            intens = torch.from_numpy(np.load(proj_path))
            intens_list.append(intens)

        self.rays_o = torch.stack(rays_o_list, 0).view(-1, 3)
        self.rays_d = torch.stack(rays_d_list, 0).view(-1, 3)
        self.intens = torch.stack(intens_list, 0).view(-1, 1)

    def __len__(self):
        return len(self.volumes) * len(self.rays_o)                                     #nr of vols * number of rays  DOES IT MAKE SENSE

    def __getitem__(self, index):                                   #index for selecting a specifc ray from rays_o/d. rays_o/d have shape 3,len(rays_o)=num_rays
      
        volume_index = index // len(self.rays_o)
        ray_index = index % len(self.rays_o)

        vol = self.volumes[volume_index]
        

        theta = self.thetas[index // self.img_size]
        inten = self.intens[index]
        ray_o, ray_d = self.rays_o[index], self.rays_d[index]

        pts = get_pts(ray_o, ray_d, self.N_samples, self.perturb)                   #get_pts gets N_samples ammount of 3d coordinates along each ray in uniform distances 
        vals = F.grid_sample(                                                       #grid_sample gets the intensity value at the points from get_pts, uses interpolation
            self.vol[None, None], pts[None, None, None], align_corners=False
        ).squeeze_()
        angle = torch.atan2(pts[..., 0], pts[..., 1])[..., None]
        radius = torch.norm(pts[..., :2], p=2, dim=-1)[..., None]
        pts = torch.cat((pts, angle, radius), dim=-1)

        return {
            "pts": pts,
            "vals": vals,
            "inten": torch.Tensor([inten]),
            "theta": torch.Tensor([theta]),
            #"object_idx": torch.Tensor([self.volume_indices[index]]),  # Add object index
       
        }

    def sample_pts(self, indexes):                                  #pts
        grid_h, grid_d, grid_w = torch.meshgrid(indexes, indexing="ij")
        pts = torch.stack([grid_w, grid_d, grid_h], -1)  # [H, D, W, 3]
        vals = F.grid_sample(                                               #HOW DOES THIS GRID SAMPLE WORK?
            self.vol[None, None], pts[None], align_corners=False
        )
        angle = torch.atan2(pts[..., 0], pts[..., 1])[..., None]
        radius = torch.norm(pts[..., :2], p=2, dim=-1)[..., None]
        pts = torch.cat((pts, angle, radius), dim=-1)

        return pts.squeeze_(), vals.squeeze_()

    def random_plane(self, view="axial"):
        if view is None:
            view = random.choice(self.valid_views)
        else:
            assert view in self.valid_views

        index_h, index_d, index_w = get_indexes(self.H, self.D, self.W)
        if view == "axial":
            index_h = torch.Tensor(
                [(2 * random.choice(range(self.H)) + 1) / self.H - 1]
            )
        elif view == "sagital":
            index_d = torch.Tensor(
                [(2 * random.choice(range(self.D)) + 1) / self.D - 1]
            )
        elif view == "coronal":
            index_w = torch.Tensor(
                [(2 * random.choice(range(self.W)) + 1) / self.W - 1]
            )

        pts, vals = self.sample_pts([index_h, index_d, index_w])
        return pts, vals

    def all_planes(self):
        pts, vals = self.sample_pts(get_indexes(self.H, self.D, self.W))
        return pts, vals

    # def tensor2vol(self, tensor):                                   #intensity -> sitk volumes
        tensor = tensor * self.args.std_val + self.args.min_val
        array = tensor.cpu().numpy()
        vol = sitk.GetImageFromArray(array)
        vol = sitk.Cast(vol, sitk.sitkInt16)
        vol.SetOrigin(self.props[1])
        vol.SetSpacing(self.props[2])
        vol.SetDirection(self.props[3])
        return vol



class RayDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N_samples = args.N_samples                             #nr of samples along each ray
        self.perturb = args.perturb                                 #ammount of perturbation to ray direction vectors?
        self.valid_views = ["axial", "sagital", "coronal"]

        vol = sitk.ReadImage(args.vol_path)                         #3d image
        self.props = [
            vol.GetSize(),
            vol.GetOrigin(),
            vol.GetSpacing(),
            vol.GetDirection(),
        ]

        vol = sitk.GetArrayFromImage(vol)
        self.vol = (torch.from_numpy(vol) - args.min_val) / args.std_val
        self.H, self.D, self.W = self.vol.shape
        self.img_size = self.H * self.W

        self.thetas = []
        self.matrixs = {}
        rays_o_list = []
        rays_d_list = []
        intens_list = []

        for ang_ind in range(args.angles):
            theta = int(ang_ind / args.angles * 180)
            self.thetas.append(theta)
            mat = rad2mat(theta / 180 * np.pi)
            self.matrixs[theta] = mat

            rays_o, rays_d = get_rays(self.H, self.W, theta, mat)
            rays_o_list.append(rays_o)
            rays_d_list.append(rays_d)

            proj_name = "{:03d}.npy".format(theta)
            proj_path = os.path.join(args.proj_dir, proj_name)
            intens = torch.from_numpy(np.load(proj_path))
            intens_list.append(intens)

        self.rays_o = torch.stack(rays_o_list, 0).view(-1, 3)
        self.rays_d = torch.stack(rays_d_list, 0).view(-1, 3)
        self.intens = torch.stack(intens_list, 0).view(-1, 1)

    def __len__(self):
        return len(self.rays_o)                                     #number of rays

    def __getitem__(self, index):
        theta = self.thetas[index // self.img_size]
        inten = self.intens[index]
        ray_o, ray_d = self.rays_o[index], self.rays_d[index]

        pts = get_pts(ray_o, ray_d, self.N_samples, self.perturb)
        vals = F.grid_sample(
            self.vol[None, None], pts[None, None, None], align_corners=False
        ).squeeze_()
        angle = torch.atan2(pts[..., 0], pts[..., 1])[..., None]
        radius = torch.norm(pts[..., :2], p=2, dim=-1)[..., None]
        pts = torch.cat((pts, angle, radius), dim=-1)

        return {
            "pts": pts,
            "vals": vals,
            "inten": torch.Tensor([inten]),
            "theta": torch.Tensor([theta]),
        }

    def sample_pts(self, indexes):                                  #pts
        grid_h, grid_d, grid_w = torch.meshgrid(indexes, indexing="ij")
        pts = torch.stack([grid_w, grid_d, grid_h], -1)  # [H, D, W, 3]
        vals = F.grid_sample(                                               #HOW DOES THIS GRID SAMPLE WORK?
            self.vol[None, None], pts[None], align_corners=False
        )
        angle = torch.atan2(pts[..., 0], pts[..., 1])[..., None]
        radius = torch.norm(pts[..., :2], p=2, dim=-1)[..., None]
        pts = torch.cat((pts, angle, radius), dim=-1)

        return pts.squeeze_(), vals.squeeze_()

    def random_plane(self, view="axial"):
        if view is None:
            view = random.choice(self.valid_views)
        else:
            assert view in self.valid_views

        index_h, index_d, index_w = get_indexes(self.H, self.D, self.W)
        if view == "axial":
            index_h = torch.Tensor(
                [(2 * random.choice(range(self.H)) + 1) / self.H - 1]
            )
        elif view == "sagital":
            index_d = torch.Tensor(
                [(2 * random.choice(range(self.D)) + 1) / self.D - 1]
            )
        elif view == "coronal":
            index_w = torch.Tensor(
                [(2 * random.choice(range(self.W)) + 1) / self.W - 1]
            )

        pts, vals = self.sample_pts([index_h, index_d, index_w])
        return pts, vals

    def all_planes(self):
        pts, vals = self.sample_pts(get_indexes(self.H, self.D, self.W))
        return pts, vals

    # def tensor2vol(self, tensor):                                   #intensity -> sitk volumes
        tensor = tensor * self.args.std_val + self.args.min_val
        array = tensor.cpu().numpy()
        vol = sitk.GetImageFromArray(array)
        vol = sitk.Cast(vol, sitk.sitkInt16)
        vol.SetOrigin(self.props[1])
        vol.SetSpacing(self.props[2])
        vol.SetDirection(self.props[3])
        return vol


class VolDataset(RayDataset):
    def __len__(self):
        return self.H * self.D * self.W

    def __getitem__(self, index):
        h = index // (self.D * self.W)
        d = (index % (self.D * self.W)) // self.W
        w = (index % (self.D * self.W)) % self.W

        val = torch.Tensor([self.vol[h, d, w]])
        h = (2 * h + 1) / self.H - 1
        d = (2 * d + 1) / self.D - 1
        w = (2 * w + 1) / self.W - 1
        pts = torch.Tensor([w, d, h])

        return {"pts": pts, "vals": val}
