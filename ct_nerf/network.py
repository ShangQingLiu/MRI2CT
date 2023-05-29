import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    """"""

    def __init__(self, input_ch, inc_input, max_freq, N_freqs):
        super().__init__()
        self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        self.out_dim = 2 * N_freqs * input_ch

        print("init embederrrrrrr dim ", self.out_dim, N_freqs, input_ch)
        if inc_input:
            self.out_dim += input_ch

        

    def forward(self, x):
        x = x * np.pi
        sin = [torch.sin(x * freq_band) for freq_band in self.freq_bands]
        cos = [torch.cos(x * freq_band) for freq_band in self.freq_bands]
        oup = torch.cat(sin + cos, -1)
        return oup



class SineLayer(nn.Module):                #is this one used?
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features, 1 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        #print("Input shape in SineLayer forward:", input.shape)
        #print("Weight shape in SineLayer forward:", self.linear.weight.shape)
        #print("Bias shape in SineLayer forward:", self.linear.bias.shape)
        
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class EmbeddingNeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,     #its actually defined as 5 when the model is called.
        latent_dim=0,
        output_ch=1,
        skips=[4],
        multi_res=8,
        matrix_size = 1,
        mode = 0,
    ):
        """ """
        super(EmbeddingNeRF, self).__init__()
        self.skips = skips
        self.embedder = Embedder(input_ch, False, multi_res - 1, multi_res) 
        self.latent_dim = latent_dim
        self.matrix_size = matrix_size
     
        
        self.mode = mode
        print("MODEEEE", self.mode)
        if self.mode == 0:

            input_ch = self.embedder.out_dim + latent_dim
        elif self.mode == 1:
            input_ch = self.embedder.out_dim 
            print("ADDITION MODE")
        else:
            print("invalid mode", self.mode)
            
        
        print("LATETNT DIMMMMMMM", latent_dim, "EMBEDDER_OUT_DIM", self.embedder.out_dim) 
        print("MODEEEEEE", mode)

        

            
       
        self.pts_linears = nn.ModuleList(                   
            [SineLayer(input_ch, W)] #input_ch + self.latent_dim for concat?
            + [
                SineLayer(W, W)
                if i not in self.skips
                else SineLayer(W + input_ch, W)  #W + input_ch + self.latent_dim for concat?
                for i in range(D - 1)
            ])

        print("latentdim, inputch,W, D ", latent_dim, input_ch, W, D)

        self.output_linear = nn.Linear(W, output_ch)            #linear transformation of input data
        with torch.no_grad():
            self.output_linear.weight.uniform_(
                -np.sqrt(6 / W) / 30,
                np.sqrt(6 / W) / 30,
            )

    
    def forward(self, x, object_latent_embedding):
        
        object_latent_matrix = object_latent_embedding.view(-1, self.latent_dim, self.matrix_size)          #tensor reshaping

        x = self.embedder(x)
        identity = x

        if self.mode == 1:  # addition mode
            for i, linear in enumerate(self.pts_linears):
                if i > 0 and i < self.matrix_size: 
                    x = x + object_latent_matrix[:, :, i-1] 
                x = linear(x)
                if i in self.skips:     # and i < self.matrix_size:
                    if i < self.matrix_size:
                        skip_input = torch.cat([identity, object_latent_matrix[:, :, i]], -1)
                    else:
                        skip_input = identity
                    x = torch.cat([skip_input, x], -1)
                    #x = torch.cat([identity, x], -1)


        elif self.mode == 0:  # concat mode
            x = torch.cat([x, object_latent_matrix[:, :, 0]], dim=-1) 
            for i, linear in enumerate(self.pts_linears):
                if i+1 < self.matrix_size:
                    if i == 0:
                        x = torch.cat([x, object_latent_matrix[:, :, i+1]], dim=-1) 
                        #i think the if i > 0 is missing
                x = linear(x)
                if i in self.skips:#and i < self.matrix_size:
                    x = torch.cat([identity, x], -1)

        outputs = self.output_linear(x)

        return outputs

    def forwardOLD(self, x, object_latent_embedding):
        x = self.embedder(x)
        print("Object latent embedding shape before view:", object_latent_embedding.shape)


        if self.mode == 0:              #concat mode
            x = torch.cat([x, object_latent_embedding], dim=-1)  # Concatenate latent_embedding with input points<----------
        
        elif self.mode == 1:            #addition mode
            x = x + object_latent_embedding


        identity = x
        for i, linear in enumerate(self.pts_linears):           #for all SineLayers
            x = linear(x)
            if i in self.skips:
                x = torch.cat([identity, x], -1)


        outputs = self.output_linear(x)

        return outputs


class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        output_ch=1,
        skips=[4],
        multi_res=8,
    ):
        """ """
        super(NeRF, self).__init__()
        self.skips = skips
        self.embedder = Embedder(input_ch, False, multi_res - 1, multi_res)
        input_ch = self.embedder.out_dim

        self.pts_linears = nn.ModuleList(                       #pts_layers is a list of SineLayer modules
            [SineLayer(input_ch, W)]
            + [
                SineLayer(W, W)
                if i not in self.skips
                else SineLayer(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        self.output_linear = nn.Linear(W, output_ch)            #linear transformation of input data
        with torch.no_grad():
            self.output_linear.weight.uniform_(
                -np.sqrt(6 / W) / 30,
                np.sqrt(6 / W) / 30,
            )

    def forward(self, x):
        x = self.embedder(x)
        identity = x
        for i, linear in enumerate(self.pts_linears):           #for all SineLayers
            x = linear(x)                                       # pass to next layer
            if i in self.skips:                                 #if its a skip layer
                x = torch.cat([identity, x], -1)

        outputs = self.output_linear(x)

        return outputs




class NeRF_incl_embedding(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        output_ch=1,
        skips=[4],
        multi_res=8,
        latent_dim = 0, #256
        case_num = 1,
        
        #base_num?
    ):
        """ """
        super(NeRF_incl_embedding, self).__init__()
        case_num = 2#case_num
        self.skips = skips
        self.embedder = Embedder(input_ch, False, multi_res - 1, multi_res)
        input_ch = self.embedder.out_dim + latent_dim     #here was the error i think
        


        self.sems = nn.Embedding(case_num, latent_dim)

        self.pts_linears = nn.ModuleList(                       #pts_layers is a list of SineLayer modules
            [SineLayer(input_ch , W)]
            + [
                SineLayer(W, W)
                if i not in self.skips
                else SineLayer(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        #print("latentdim, inputch,W, D ", latent_dim, input_ch, W, D)  #WHY THE FUCK IS W= 5 and D=8?????

        self.output_linear = nn.Linear(W, output_ch)            #linear transformation of input data
        with torch.no_grad():
            self.output_linear.weight.uniform_(
                -np.sqrt(6 / W) / 30,
                np.sqrt(6 / W) / 30,
            )

    def forward(self, x, idx):

        sems = self.sems(idx)

      
        x = self.embedder(x)
        sems = sems.squeeze()

        N = 64   #num ryays
        sems = sems.repeat_interleave(N, dim=0)
        # Concatenate x and sems
        x = torch.cat([x, sems], -1)
        
        identity = x
        for i, linear in enumerate(self.pts_linears):           #for all SineLayers

            #print(f"Input shape at layer {i}: {x.shape}")
            x = linear(x)
            if i in self.skips:
                x = torch.cat([identity, x], -1)

       

        outputs = self.output_linear(x)

        return outputs

    

class GaussinEmbedder(nn.Module):       #not used??
    """"""

    def __init__(self, input_ch, output_ch=128):
        super().__init__()
        self.B = nn.Linear(input_ch, output_ch, bias=False)
        with torch.no_grad():
            self.B.weight.normal_()
        self.B.requires_grad_(False)
        self.out_dim = output_ch * 2

    def forward(self, x):
        x = 2 * x * torch.pi
        x = self.B(x)
        oup = torch.cat([torch.sin(x), torch.cos(x)], -1)
        return oup



if __name__ == "__main__":
    from .parser import config_parser

    parser = config_parser()
    args = parser.parse_args()

    model = NeRF()
    x = torch.zeros(5, 3)
    y = model(x)
