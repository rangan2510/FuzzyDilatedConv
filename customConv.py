#%%
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch.nn.modules.utils import _pair

#%%
class customConv(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1, bias=True, mu=0.9):
        super(customConv, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.out_channels = out_channels
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        self.n_channels = n_channels
        self.mu = mu
        self.weight = Parameter(torch.Tensor(self.out_channels, self.n_channels, self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.n_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_):

        hout = ((input_.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0]-1)-1)//self.stride[0])+1
        wout = ((input_.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1]-1)-1)//self.stride[1])+1
        mu = 0.9
        _mu = (1-mu)/(3)
        neighbor_weight_mask = torch.tensor([[0, _mu, 0, _mu, 0], [_mu, _mu, _mu, _mu, _mu], [0, _mu, 0, _mu, 0], [_mu, _mu, _mu, _mu, _mu],[0, _mu, 0, _mu, 0]])
        dilator_mask = torch.tensor([[1., 0., 1., 0., 1.], [0., 0., 0., 0., 0.], [1., 0., 1., 0., 1.], [0., 0., 0., 0., 0.],[1., 0., 1., 0., 1.]])

        dilated_kernel = self.weight[:]*dilator_mask
        f_dilated_kernel = dilated_kernel[:] + neighbor_weight_mask
        
        inputUnfolded = F.unfold(input_, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride)
        
        convolvedOutput = (inputUnfolded.transpose(1, 2).matmul(f_dilated_kernel.view(self.weight.size(0), -1).t()).transpose(1, 2))


        convolutionReconstruction = convolvedOutput.view(input_.shape[0], self.out_channels, hout, wout)
        return convolutionReconstruction

