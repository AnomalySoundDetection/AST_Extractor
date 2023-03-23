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
    def __init__(
        self,
        backbone_name,
        flow_steps,
        # input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,
    ):
        super(FastFlow, self).__init__()
        
        #assert (
        #    backbone_name in const.SUPPORTED_BACKBONES
        #), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)
        #self.backbone_name = backbone_name
        
        #if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
        #    self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
        #    channels = [768]
        #    scales = [16]
        #elif backbone_name in [const.BACKBONE_RESNET18, const.BACKBONE_WIDE_RESNET50]:
        #    self.feature_extractor = timm.create_model(
        #        backbone_name,
        #        pretrained=True,
        #        features_only=True,
        #        out_indices=[1, 2, 3],
        #    )
            # *************
            # pretrained_dict = torch.load('pretrained_chechpoint/pre-trained/39.pt')["model_state_dict"]
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.feature_extractor.state_dict()}
            # self.feature_extractor.load_state_dict(pretrained_dict)
            # *************
        #    channels = self.feature_extractor.feature_info.channels()
        #    scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
        #    self.norms = nn.ModuleList()
        #    for in_channels, scale in zip(channels, scales):
        #        self.norms.append(
        #            nn.LayerNorm(
        #                [in_channels, int(input_size / scale), int(input_size / scale)],
        #                elementwise_affine=True,
        #            )
        #        )
        #else:
        #    self.feature_extractor = load_extractor()
        #    self.norms = nn.ModuleList()
        #    channels = [64, 128, 256]
        #    sizes = [[625, 32], [312, 16], [156, 8]]
        #    for in_channels, size in zip(channels, sizes):
        #        self.norms.append(
        #            nn.LayerNorm(
        #                [in_channels, size[0], size[1]],
        #                elementwise_affine=True,
        #            )
        #        )




        #for param in self.feature_extractor.parameters():
        #    param.requires_grad = False

        #self.nf_flows = nn.ModuleList()
        #if self.backbone_name in [const.BACKBONE_DEIT, const.BACKBONE_CAIT, const.BACKBONE_RESNET18, const.BACKBONE_WIDE_RESNET50]:
        '''
        channel: 768
        size of feature: 55 * 22
        '''
        
        self.nf_flows.append(
            NF_Fast_Flow(
                [768, 55, 22], 
                conv3x3_only=conv3x3_only,
                hidden_ratio=hidden_ratio,
                flow_steps=flow_steps,    
            )
        )

        #for in_channels, scale in zip(channels, scales):
        #    self.nf_flows.append(
        #        Nf_fast_flow(
        #            [in_channels, int(input_size / scale), int(input_size / scale)],
        #            conv3x3_only=conv3x3_only,
        #            hidden_ratio=hidden_ratio,
        #            flow_steps=flow_steps,
        #        )
        #    )
        #else:
        #    for in_channels, size in zip(channels, sizes):
        #        self.nf_flows.append(
        #            nf_fast_flow(
        #                [in_channels, size[0], size[1]],
        #                conv3x3_only=conv3x3_only,
        #                hidden_ratio=hidden_ratio,
        #                flow_steps=flow_steps,
        #            )
        #        )
        # self.input_size = input_size

    def forward(self, x):
        #self.feature_extractor.eval()
        #if isinstance(
        #    self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        #):
        #    x = self.feature_extractor.patch_embed(x)
        #    cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
        #    if self.feature_extractor.dist_token is None:
        #        x = torch.cat((cls_token, x), dim=1)
        #    else:
        #        x = torch.cat(
        #            (
        #                cls_token,
        #                self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
        #                x,
        #            ),
        #            dim=1,
        #        )
        #    x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        #    for i in range(8):  # paper Table 6. Block Index = 7
        #        x = self.feature_extractor.blocks[i](x)
        #    x = self.feature_extractor.norm(x)
        #    x = x[:, 2:, :]
        #    N, _, C = x.shape
        #    x = x.permute(0, 2, 1)
        #    x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
        #    features = [x]
        #elif isinstance(self.feature_extractor, timm.models.cait.Cait):
        #    x = self.feature_extractor.patch_embed(x)
        #    x = x + self.feature_extractor.pos_embed
        #    x = self.feature_extractor.pos_drop(x)
        #    for i in range(41):  # paper Table 6. Block Index = 40
        #        x = self.feature_extractor.blocks[i](x)
        #    N, _, C = x.shape
        #    x = self.feature_extractor.norm(x)
        #    x = x.permute(0, 2, 1)
        #    x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
        #    features = [x]
        #elif self.backbone_name in [const.BACKBONE_RESNET18, const.BACKBONE_WIDE_RESNET50]:
        #    features = self.feature_extractor(x)
        #    features = [self.norms[i](feature) for i, feature in enumerate(features)]
            # print(features[0].shape)
            # print(features[1].shape)
            # print(features[2].shape)
            # exit()
        #else:
            # not done: extract certain layers' outputs
            # self.feature_extractor(x)
        #    with torch.no_grad():
                
        #        f1, f2, f3 = self.feature_extractor(x)
        #        features = [f1, f2, f3]
                # def hook(module, input, output): 
                #     features.append(output.detach())
                # handle = self.feature_extractor.resnet.layer1.register_forward_hook(hook)
                # handle = self.feature_extractor.resnet.layer2.register_forward_hook(hook)
                # handle = self.feature_extractor.resnet.layer3.register_forward_hook(hook)
                # y = self.feature_extractor(x)
                # print(len(features))
                # print(features[0].shape)
                # print(features[1].shape)
                # print(features[2].shape)
                # handle.remove()

        #        features = [self.norms[i](feature) for i, feature in enumerate(features)]
        #        del f1, f2, f3
        
        #loss = 0
        #outputs = []
        output, log_jac_dets = self.nf_flows(x)

        return output, log_jac_dets

        #ret = {"loss": loss}

        
        #if not self.training:
        #    pr = 0.05
        #    anomaly_map_list = []
        #    for output in outputs:
        #        log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
        #        prob = torch.exp(log_prob)
        #        
        #        prob_1d = prob.cpu().detach().numpy()
        #        prob_1d = np.reshape(prob_1d, (prob_1d.shape[0], 1, prob_1d.shape[2]*prob_1d.shape[3]))
        #        anomaly_score = []
        #        for p_1d in prob_1d:
        #            anomaly_score.append([[np.mean(np.sort(p_1d, axis=None)[:int(p_1d.shape[1]*pr)])]])
        #            # print(np.sort(p_1d, axis=None)[:10])
        #            # exit()
        #        anomaly_score  = -torch.tensor(anomaly_score)
                # a_map = F.interpolate(
                #     -prob,
                #     size=[500, 500],
                #     mode="bilinear",
                #     align_corners=False,
                # )
                # print(a_map.shape)
                # print(output.shape)
                # print(prob_1d.shape)
                # print(anomaly_score.shape)
                # exit()
        #        anomaly_map_list.append(anomaly_score)
        #    anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
        #    anomaly_map = torch.mean(anomaly_map_list, dim=-1)
        #    ret["anomaly_map"] = anomaly_map
        #return ret