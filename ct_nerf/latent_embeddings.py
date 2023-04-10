# latent_embeddings.py

import torch
import math

#does it have to be a CLASS? i guess not

def init_latent_embeddings(num_patients, latent_size, init_std_dev=1.0, max_norm=None):
    lat_vecs = torch.nn.Embedding(num_patients, latent_size, max_norm=max_norm)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        init_std_dev / math.sqrt(latent_size),
    )
    return lat_vecs
