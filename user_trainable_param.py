#%%
import torch
import torch.nn as nn

#%%
class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        self.weight = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, 1), requires_grad=True)
        
        self.weight.data.uniform_(-1, 1)
        
    
    def forward(self, x):
        masked_wt = self.weight.mul(1)
        return masked_wt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Mask = Mask()

    def forward(self,x):
        x = Mask(x)
        return x

model = Model()

# %%
