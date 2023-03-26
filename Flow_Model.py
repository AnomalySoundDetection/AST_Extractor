import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
#import constants as const
import numpy as np

def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def NF_Fast_Flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )

    return nodes


class FastFlow(nn.Module):
    def __init__(self, flow_steps, conv3x3_only=False, hidden_ratio=1.0):
        super(FastFlow, self).__init__()

        self.nf_flows = nn.ModuleList()
        self.nf_flows.append(NF_Fast_Flow([768, 12, 101], 
                                conv3x3_only=conv3x3_only, 
                                hidden_ratio=hidden_ratio, 
                                flow_steps=flow_steps)
                            )
        

        #print(self.nf_flows)


    def forward(self, x):
        output, log_jac_dets = self.nf_flows[0](x)
        
        return output, log_jac_dets
