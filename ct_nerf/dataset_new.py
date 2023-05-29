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
        # self.mode = mode


        # CHANGE: Added self.volume_indices
        self.volume_indices = []     #length = total nr of rays, keeps track of which volume index each ray belongs to. 


  
        volume_paths = [os.path.join(args.volumes_dir, vol) for vol in os.listdir(args.volumes_dir)]

        #for UPDATE.py loading a specific file for finetuning
        if args.finetune_file is not None:
            print("Finetune and optimize LE on", args.finetune_file)
            volume_paths = [vol_path for vol_path in volume_paths if args.finetune_file in vol_path]
            args.num_vols = 1 #not nessecarily needed


        #load and store the multiple volumes in lists:
        self.vol_list = []               #contains all 3d volumes loaded (before self.volumes)
        self.props_list = []                 #list of properties for each volume
        self.H_list = []
        self.D_list = []
        self.W_list = []
        self.img_size_list = []

        #num_vol = args.num_vols

        #for vol_path in volume_paths:
        for i, vol_path in enumerate(volume_paths[:args.num_vols]):
            print("processing how many volumes?", args.num_vols)

            vol = sitk.ReadImage(args.vol_path)                         #3d image
            print(f"Processing file: {vol_path}") 
            self.props = [ #or self.props.append([
                vol.GetSize(),
                vol.GetOrigin(),
                vol.GetSpacing(),
                vol.GetDirection(),
            ]

            #self.props.append(vol_props)  # Append the properties of the current volume to the list, adress with self.props[volume_index]


            vol = sitk.GetArrayFromImage(vol)
            self.vol = (torch.from_numpy(vol) - args.min_val) / args.std_val            #normalization
            self.H, self.D, self.W = self.vol.shape
            self.img_size = self.H * self.W


            #append everything to the lists
            self.vol_list.append(self.vol)
            self.props_list.append(self.props)
            self.H_list.append(self.H)
            self.D_list.append(self.D)
            self.W_list.append(self.W)
            self.img_size_list.append(self.img_size)
            
            self.total_volumes = len(self.vol_list)
            print("totalvols:", self.total_volumes, "imgsizelist", self.H_list)

            #waht
            #self.volume_indices += [len(self.vol_list) - 1] * self.img_size_list[-1]
            print("volindeces", self.volume_indices) #72704 * 0
            self.total_samples = sum(self.img_size_list)
            

            #self.volume_indices = [i for i, img_size in enumerate(self.img_size_list) for _ in range(img_size)]
            #self.total_samples = len(self.volume_indices)
            print(f"Length of volume_indices: {len(self.volume_indices)}")

            

            print(f"total samples: {self.total_samples}")

        

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


            #list of lists for rays and intens
            rays_o_list_vol = []
            rays_d_list_vol = []
            intens_list_vol = []

            # CHANGE: Loop through volumes and populate rays_o_list_vol, rays_d_list_vol, intens_list_vol
            for vol_idx in range(self.total_volumes):
                H, W = self.H_list[vol_idx], self.W_list[vol_idx]

                rays_o, rays_d = get_rays(H, W, theta, mat)
                rays_o_list_vol.append(rays_o)
                rays_d_list_vol.append(rays_d)

                proj_name = "{:03d}.npy".format(theta)
                proj_path = os.path.join(args.proj_dir, proj_name)
                intens = torch.from_numpy(np.load(proj_path))
                intens_list_vol.append(intens)

                self.volume_indices += [len(self.vol_list) - 1] * self.img_size_list[-1]

                
         # CHANGE: Stack and concatenate rays_o_list_vol, rays_d_list_vol, intens_list_vol for all volumes
            rays_o_list.append(torch.stack(rays_o_list_vol, 0).view(-1, 3))
            rays_d_list.append(torch.stack(rays_d_list_vol, 0).view(-1, 3))
            intens_list.append(torch.stack(intens_list_vol, 0).view(-1, 1))
        
        print(f"Length of rays_o_list: {len(rays_o_list)}")   #180???
        
        self.rays_o = torch.cat(rays_o_list, 0)
        self.rays_d = torch.cat(rays_d_list, 0)
        self.intens = torch.cat(intens_list, 0)

        print(f"Length of rays_o: {len(rays_o)}")
        print("len thetas", len(self.thetas))


    def __len__(self):
        #return total_length
        return self.rays_o.shape[0]

    def __getitem0__(self, index):                                   #index for selecting a specifc ray from rays_o/d. rays_o/d have shape 3,len(rays_o)=num_rays
      
        vol_idx = index // sum(self.img_size_list)
        vol_img_size_sum = sum(self.img_size_list[:vol_idx])
        vol_index = index - vol_img_size_sum
        
        print(f"vol_idx: {vol_idx}, self.img_size: {self.img_size_list[:vol_idx]}, vol_index: {len(vol_index)}")
        #print(f"Index: {index}, self.img_size: {self.img_size}, len(self.thetas): {len(self.thetas)}")
    
        theta = self.thetas[vol_index // self.img_size_list[vol_idx]]                     #ray_index?
        inten = self.intens[index]                                      #ray_index?
        ray_o, ray_d = self.rays_o[index], self.rays_d[index]           #ray_index?

        pts = get_pts(ray_o, ray_d, self.N_samples, self.perturb)                   #get_pts gets N_samples ammount of 3d coordinates along each ray in uniform distances  #grid_sample gets the intensity value at the points from get_pts, uses interpolation
        #vals = F.grid_sample(self.vol[None, None], pts[None, None, None], align_corners=False).squeeze_()
        vals = F.grid_sample(self.vol_list[vol_idx][None, None], pts[None, None, None], align_corners=False).squeeze_()

        angle = torch.atan2(pts[..., 0], pts[..., 1])[..., None]
        radius = torch.norm(pts[..., :2], p=2, dim=-1)[..., None]
        pts = torch.cat((pts, angle, radius), dim=-1)

        return {
            "pts": pts,
            "vals": vals,
            "inten": torch.Tensor([inten]),
            "theta": torch.Tensor([theta]),
            
            #"object_idx": torch.tensor([volume_index]).unsqueeze(0),
            "object_idx": torch.Tensor([self.volume_indices[index]]),  # Add object index

            #return {"pts": pts, "vals": vals, "object_idx": torch.tensor(patient_idx).unsqueeze(0)}

       
        }

    def __getitem__(self, index):
        # Find the volume index associated with the current index
        
        #print("lenvolumeindices", len(self.volume_indices))
        vol_idx = self.volume_indices[index]

        # Find the index within the current volume
        vol_index = index % self.img_size_list[vol_idx]

        theta = self.thetas[vol_index // self.img_size_list[vol_idx]]
        inten = self.intens[index]
        ray_o, ray_d = self.rays_o[index], self.rays_d[index]

        pts = get_pts(ray_o, ray_d, self.N_samples, self.perturb)
        vals = F.grid_sample(
            self.vol_list[vol_idx][None, None], pts[None, None, None], align_corners=False
        ).squeeze_()
        angle = torch.atan2(pts[..., 0], pts[..., 1])[..., None]
        radius = torch.norm(pts[..., :2], p=2, dim=-1)[..., None]
        pts = torch.cat((pts, angle, radius), dim=-1)

        #print("returned object idx",torch.Tensor([self.volume_indices[index]]))

        return {
            "pts": pts,
            "vals": vals,
            "inten": torch.Tensor([inten]),
            "theta": torch.Tensor([theta]),
            "object_idx": torch.Tensor([self.volume_indices[index]]),
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
