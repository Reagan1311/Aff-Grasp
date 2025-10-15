import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_util import *

# peft
from peft import LoraConfig, inject_adapter_in_model

from models.dino.vit_adapter import vit_base, vit_small, vit_large
from data.ego_video_data import AFF_LIST

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.class_names = AFF_LIST
        self.num_aff = len(self.class_names)
        self.dino_dim = 768

        # DINOv2 with LoRA
        lora_config = LoraConfig(
            target_modules=['qkv'], inference_mode=False, r=8, lora_alpha=4, lora_dropout=0.1
        )

        # self.aff_protos = nn.Parameter(torch.load('./aff-vitl-14.pt')).float()
        self.aff_protos = nn.Parameter(torch.zeros(len(AFF_LIST), self.dino_dim))
        nn.init.normal_(self.aff_protos, std=0.02)

        dino_model = vit_base(img_size=518, patch_size=14, num_register_tokens=0, block_chunks=0,
                                   init_values=1).cuda()
        state_dict = torch.load('dinov2_vitb14_pretrain.pth', map_location='cpu')

        dino_model.load_state_dict(state_dict, strict=False)
        
        self.dino_model = dino_model
        self.dino_model = inject_adapter_in_model(lora_config, dino_model)

        self.embedder = Mlp(in_features=self.dino_dim, hidden_features=int(self.dino_dim*4), out_features=self.dino_dim,
                            act_layer=nn.GELU, drop=0.)
        self.depth_embedder = Mlp(in_features=self.dino_dim, hidden_features=int(self.dino_dim*4), out_features=self.dino_dim,
                        act_layer=nn.GELU, drop=0.)
        
        self._freeze_stages(exclude_key=['aff_protos', 'embedder', 'depth', 'injectors', 'lora', 'depth_embedder'])

    def forward(self, img, depth=None, label=None, stage2=False):

        # {
        #     "x_norm_clstoken": x_norm[:, 0],
        #     "x_norm_patchtokens": x_norm[:, 1:],
        #     "x_prenorm": x,
        #     "masks": masks,
        # }

        b, _, h, w = img.shape
        patch_size = h // 14

        dino_dense = self.dino_model.forward_features(img, depth=depth)['x_norm_patchtokens']            
        dino_dense = self.embedder(dino_dense)

        dino_up = dino_dense.permute(0, 2, 1).reshape(b, -1, patch_size, patch_size)
        dino_up = F.interpolate(dino_up, scale_factor=4, mode='bilinear', align_corners=False)
        dino_up_flat = dino_up.flatten(2)
        dino_up_flat = dino_up_flat / dino_up_flat.norm(dim=1, keepdim=True)

        aff_ps = self.aff_protos / self.aff_protos.norm(dim=-1, keepdim=True)
        pred = (aff_ps.unsqueeze(0) @ dino_up_flat).reshape(b, -1, patch_size * 4, patch_size * 4)
        pred = F.interpolate(pred, size=img.shape[-2:], mode='bilinear', align_corners=False)
        pred = pred.clamp(min=0.0001, max=0.9999)

        if label != None:
            label = label[:, 1:]
            loss_bce = sigmoid_focal_loss(pred, label, alpha=-1)
            loss_dice = softdice_loss(pred, label)
            loss_dict = {'focal': loss_bce, 'dice': loss_dice}

            return pred, loss_dict

        else:
            return pred


    def _freeze_stages(self, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in self.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count > 0:
                        m.requires_grad = True
                        # print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
