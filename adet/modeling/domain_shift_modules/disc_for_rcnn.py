import torch
from torch import nn
from .grad_modules import GradReverse
import torch.nn.functional as F
from adet.modeling.distribution_measure import (
    build_combined_batch_loss,
    build_combined_batch_and_single_loss,
)

from .dat import StrideModule


__all__ = [
    "FusionDiscriminator",
    "StudentAccusingDiscriminator",
]


class FusionDiscriminator(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.fuse_type = cfg.FUSION_SETTING.FUSE_TYPE
        if self.fuse_type == 'add':
            assert len(cfg.FUSION_SETTING.ADD_AVG_WEIGHTS)==3, "cfg.FUSION_SETTING.ADD_AVG_RATIO must be length of 3"
            self.fuse_weights = cfg.FUSION_SETTING.ADD_AVG_WEIGHTS

        self.fore_backbones = self.build_fore_backbones(cfg)
        self.fore_encoder = self.build_fore_encoder(cfg)

        self.post_backbones = self.build_post_backbones(cfg)
        self.post_encoder = self.build_post_encoder(cfg)

        self.output_backbones = self.build_output_backbones(cfg)
        self.output_encoder = self.build_output_encoder(cfg)

        self.classifier = self.build_classifier(cfg)

        self.disc_loss_fn = nn.BCELoss()

    def build_fore_backbones(self, cfg):
        p2 = nn.Sequential(StrideModule(256, channel_division=5, stride_n=4), nn.Conv2d(8,8,3,padding=[1,0]))
        p3 = nn.Sequential(StrideModule(256, channel_division=5, stride_n=3), nn.Conv2d(8,8,3,padding=[1,0]))
        p4 = nn.Sequential(StrideModule(256, channel_division=5, stride_n=2), nn.Conv2d(8,8,3,padding=[1,0]))
        p5 = nn.Sequential(StrideModule(256, channel_division=5, stride_n=1), nn.Conv2d(8,8,3,padding=[1,0]))
        p6 = nn.Sequential(StrideModule(256, channel_division=5, stride_n=0), nn.Conv2d(8,8,3,padding=[1,0]))

        # p2 = StrideModule(256, channel_division=5, stride_n=4)
        # p3 = StrideModule(256, channel_division=5, stride_n=3)
        # p4 = StrideModule(256, channel_division=5, stride_n=2)
        # p5 = StrideModule(256, channel_division=5, stride_n=1)
        # p6 = StrideModule(256, channel_division=5, stride_n=0)

        return nn.ModuleList([p2, p3, p4, p5, p6])
    
    def build_fore_encoder(self, cfg):
        module = nn.Sequential(
            nn.Conv2d(40, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        return module
    
    def build_post_backbones(self, cfg):
        modules = [
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(16),
            )
        for _ in range(3)]

        return nn.ModuleList(modules)
    
    def build_post_encoder(self, cfg):
        module = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        )
        return module
    
    def build_output_backbones(self, cfg):
        modules = [
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=2),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
            )
            for _ in range(2)
        ]
        return nn.ModuleList(modules)
    
    def build_output_encoder(self, cfg):
        module = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        )
        return module
    
    def build_classifier(self, cfg):
        if self.fuse_type == 'concat':
            in_chn = 64*3
        elif self.fuse_type == 'add':
            in_chn = 64
        else:
            raise ValueError(f"Unknown fuse type \"{self.fuse_type}\"")
        module = nn.Sequential(
            nn.Conv2d(in_chn, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        return module

    def fuse(self, fore_features, post_features, output_masks):
        if self.fuse_type == 'concat':
            fused = torch.cat([fore_features, post_features, output_masks], dim=1)
        elif self.fuse_type == 'add':
            fused = (fore_features*self.fuse_weights[0] + post_features*self.fuse_weights[1] + output_masks*self.fuse_weights[2])
        else:
            raise ValueError(f"Unknown fuse type \"{self.fuse_type}\"")
        return fused

    def _sub_forward(self, x, backbones, encoder):
        features = []
        for i in range(len(x)):
            feature = GradReverse.apply(x[i])    # Negative sign for inversing gradient
            feature = backbones[i](feature)
            features.append(feature)
        features = torch.cat(features, dim=1)
        features = encoder(features)
        return features

    def forward(self,
        fore_features,  # Shapes:   (1, 256, 120, 160)
                        #           (1, 256, 60, 80)
                        #           (1, 256, 30, 40)
                        #           (1, 256, 15, 20)
                        #           (1, 256, 8, 10)
        post_features,  # [N,256,14,14] * 3
        output_masks,   # {'visible': (N, 1, 28, 28),
                        #  'amodal':  (N, 1, 28, 28) }
        domain_is_src: bool,
    ):
        assert fore_features[0].shape[0]==1, f"Sorry, currently only supports batch_size of 1, while appears to have batch_size={fore_features[0].shape[0]}."
        assert len(fore_features)==5, f"\"fore_features\" length should be 5, but it's {len(fore_features)}"
        fore_features = self._sub_forward(fore_features, self.fore_backbones, self.fore_encoder)
        # print("fore_features", fore_features.shape)

        assert len(post_features)==3, f"Expect \"post_features\" to have length of 3, appears to be {len(post_features)}."
        if post_features[0].shape[0] == 0:
            return 0.0
        post_features = self._sub_forward(post_features, self.post_backbones, self.post_encoder)
        # print("post_features", post_features.shape)
        
        assert len(output_masks)==2, f"Expect \"output_masks\" to have length of 2, appears to be {len(output_masks)}."
        output_masks = [output_masks['visible'], output_masks['amodal']]
        output_masks = self._sub_forward(output_masks, self.output_backbones, self.output_encoder)
        # print("output_masks", output_masks.shape)

        dim_target = post_features.shape[0]
        depth = fore_features.shape[1:]
        fore_features = fore_features.expand(dim_target, *depth)

        classifier_input = self.fuse(fore_features, post_features, output_masks)
        # print("classifier_input", classifier_input.shape)
        domain_pred = self.classifier(classifier_input)
        # print("domain_pred", domain_pred.shape)

        assert isinstance(domain_is_src, bool), "\"domain_is_src\" must be bool"
        if domain_is_src is True:
            gt_domain_pred = torch.ones_like(domain_pred)
        else:
            gt_domain_pred = torch.zeros_like(domain_pred)
        
        loss = F.binary_cross_entropy(domain_pred, gt_domain_pred)
        
        return loss


class StudentAccusingDiscriminator(nn.Module):
    """
    InputRoI[B,256,14,14]   ->   VisFeat[B,256,14,14]
                                 └-> VisOut[B,1,28,28]
                                 └-> Disc
                        └---└->  AmoFeat[B,256,14,14]
                                 └-> AmoOut[B,1,28,28]
                                 └-> Disc
    """
    def __init__(self, cfg) -> None:
        super().__init__()

        self.use_resnet = cfg.DOMAIN_ADAPTATION.USE_RESNET
        self.do_disc_on_gt = cfg.DOMAIN_ADAPTATION.STUDENT_DISC_ON_GT_OUTMASK
        self.dropout_rate = cfg.DOMAIN_ADAPTATION.STUDENT_DROPOUT

        self.vis_feat_module = self.build_feat_module(cfg, in_chn_n=256)
        self.vis_out_module = self.build_out_module(cfg)
        self.vis_disc = self.build_out_disc(cfg)

        self.amo_feat_module = self.build_feat_module(cfg, in_chn_n=512)
        self.amo_out_module = self.build_out_module(cfg)
        self.amo_disc = self.build_out_disc(cfg)
        
        # self.distil_loss_fn = build_combined_batch_loss()
        self.distil_loss_fn = build_combined_batch_and_single_loss()

        self.lrn_w_visfeat: float = cfg.DOMAIN_ADAPTATION.LRN_WEIGHT_VISFEAT
        self.lrn_w_amofeat: float = cfg.DOMAIN_ADAPTATION.LRN_WEIGHT_AMOFEAT
        self.lrn_w_visout: float = cfg.DOMAIN_ADAPTATION.LRN_WEIGHT_VISOUT
        self.lrn_w_amoout: float = cfg.DOMAIN_ADAPTATION.LRN_WEIGHT_AMOOUT
        self.acs_w_roifeat: float = cfg.DOMAIN_ADAPTATION.ACS_WEIGHT_ROIFEAT
        self.acs_w_visfeat: float = cfg.DOMAIN_ADAPTATION.ACS_WEIGHT_VISFEAT
        self.acs_w_amofeat: float = cfg.DOMAIN_ADAPTATION.ACS_WEIGHT_AMOFEAT
        self.acs_w_visout: float = cfg.DOMAIN_ADAPTATION.ACS_WEIGHT_VISOUT
        self.acs_w_amoout: float = cfg.DOMAIN_ADAPTATION.ACS_WEIGHT_AMOOUT
        self.hard_out_mask: bool = cfg.DOMAIN_ADAPTATION.STUDENT_HARD_MASK
        self.bce_on_mask: bool = cfg.DOMAIN_ADAPTATION.STUDENT_BCE_ON_MASK

    def build_feat_module(self, cfg, in_chn_n):
        module = nn.Sequential()
        kernel_sizes = [7,5,3]
        if self.use_resnet is False:
            for i, knl_sz in enumerate(kernel_sizes):
                chn_n = in_chn_n if i==0 else 256
                module.add_module(f"l{i}_0", nn.Conv2d(chn_n, 256, knl_sz, padding=((knl_sz-1)//2)))
                module.add_module(f"l{i}_1", nn.LeakyReLU())
                module.add_module(f"l{i}_2", nn.Dropout2d(0.12))
        else:
            if in_chn_n != 256:
                module.add_module(f"l_init", nn.Conv2d(in_chn_n, 256, 3, padding=1))
            for i, knl_sz in enumerate(kernel_sizes):
                module.add_module(f"l{i}", _ResNetBlock(256, knl_sz, padding=((knl_sz-1)//2), add_bn=False, drop_out=self.dropout_rate))
        module.add_module(f"l{i+1}", nn.BatchNorm2d(256))
        return module

    def build_out_module(self, cfg):
        if self.use_resnet is False:
            module = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(256, 32, 2, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(32, 1, 1),
            )
        else:
            module = nn.Sequential(
                _ResNetBlock(256, 3, padding=1, add_bn=False),
                _ResNetBlock(256, 3, padding=1, add_bn=False),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(256, 32, 2, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(32, 1, 1),
            )
        return module

    def build_out_disc(self, cfg):
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # [N,16,14,14]
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),   # [N,16,7,7]
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),   # [N,16,7,7]
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=4),    # [N,16,28,28]
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, kernel_size=1),    # [N,1,28,28]
        )

    def forward(
        self,
        roi_input: torch.Tensor,
        vis_feat_gt: torch.Tensor,
        vis_out_gt: torch.Tensor,
        amo_feat_gt: torch.Tensor,
        amo_out_gt: torch.Tensor,
        domain_is_src: bool,
        operation: str,
    ):
        if operation == 'learn_and_judge':
            if (self.hard_out_mask is True) and (vis_out_gt is not None) and (amo_out_gt is not None):
                vis_out_gt = (vis_out_gt>0).float()
                amo_out_gt = (amo_out_gt>0).float()
            
            if roi_input.shape[0] == 0:
                return {'stuLrn_loss': 0.0,'stuDisc_loss': 0.0}

            ### Learn
            learn_loss, vis_out, amo_out = self._learn(
                roi_input, vis_feat_gt, vis_out_gt, amo_feat_gt, amo_out_gt,
            )
            
            ### Self discriminate
            disc_loss = self._judge(vis_out, amo_out, domain_is_src=domain_is_src)
            if self.do_disc_on_gt is True:
                disc_loss += self._judge(vis_out_gt, amo_out_gt, domain_is_src=domain_is_src)
                disc_loss /= 2
            return {
                'stuLrn_loss': learn_loss,
                'stuDisc_loss': disc_loss,
            }
        
        elif operation == 'accuse':
            if roi_input.shape[0] == 0:
                return {'stuAcs_loss': 0.0}
            loss = self._accuse(roi_input, vis_feat_gt, vis_out_gt, amo_feat_gt, amo_out_gt, domain_is_src=domain_is_src)
            return {'stuAcs_loss': sum(loss.values())}
        else:
            raise ValueError(f"Unrecognized operation=\"{operation}\"")

    def _learn(
        self,

        roi_input: torch.Tensor,

        vis_feat_gt: torch.Tensor,
        vis_out_gt: torch.Tensor,
        amo_feat_gt: torch.Tensor,
        amo_out_gt: torch.Tensor,
    ):
        vis_branch_input = roi_input
        vis_feat = self.vis_feat_module(vis_branch_input)   # -> [B,256,14,14]
        vis_out = self.vis_out_module(vis_feat)

        amo_branch_input = torch.cat([roi_input, vis_feat], dim=1)
        amo_feat = self.amo_feat_module(amo_branch_input)   # -> [B,256,14,14]
        amo_out = self.amo_out_module(amo_feat)

        vis_feat_loss = self.distil_loss_fn(vis_feat, vis_feat_gt)
        amo_feat_loss = self.distil_loss_fn(amo_feat, amo_feat_gt)
        
        if (vis_out_gt is not None) and (amo_out_gt is not None):
            vis_out_loss = self.distil_loss_fn(vis_out, vis_out_gt) if (self.bce_on_mask is False) else F.binary_cross_entropy(vis_out.sigmoid(), vis_out_gt)
            amo_out_loss = self.distil_loss_fn(amo_out, amo_out_gt) if (self.bce_on_mask is False) else F.binary_cross_entropy(amo_out.sigmoid(), amo_out_gt)
        else:
            vis_out_loss = 0.0
            amo_out_loss = 0.0

        loss = self.lrn_w_visfeat * vis_feat_loss + \
               self.lrn_w_amofeat * vis_out_loss + \
               self.lrn_w_visout  * amo_feat_loss + \
               self.lrn_w_amoout  * amo_out_loss

        return loss, vis_out, amo_out

    def _judge(
        self,
        vis_out: torch.Tensor,
        amo_out: torch.Tensor,
        domain_is_src: bool,
    ):
        vis_out = GradReverse.apply(vis_out)
        amo_out = GradReverse.apply(amo_out)

        vis_disc = self.vis_disc(vis_out)
        amo_disc = self.amo_disc(amo_out)

        domain_gt = torch.ones_like(vis_disc) if domain_is_src==True else torch.zeros_like(vis_disc)
        vis_disc_loss = F.binary_cross_entropy(vis_disc.sigmoid(), domain_gt)
        amo_disc_loss = F.binary_cross_entropy(amo_disc.sigmoid(), domain_gt)

        loss = vis_disc_loss + amo_disc_loss
        return loss

    def _accuse(
        self,
        roi_input: torch.Tensor,
        vis_feat_gt: torch.Tensor,
        vis_out_gt: torch.Tensor,
        amo_feat_gt: torch.Tensor,
        amo_out_gt: torch.Tensor,
        domain_is_src: bool,
    ):

        domain_gt = None
        loss = {
            'roi_input': 0.0,
            'vis_feat': 0.0,
            'vis_out': 0.0,
            'amo_feat': 0.0,
            'amo_out': 0.0,
        }

        if self.acs_w_roifeat > 0.0:
            roi_input = GradReverse.apply(roi_input)
            vis_branch_input = roi_input
            vis_feat = self.vis_feat_module(vis_branch_input)
            vis_out = self.vis_out_module(vis_feat)
            amo_branch_input = torch.cat([roi_input, vis_feat], dim=1)
            amo_feat = self.amo_feat_module(amo_branch_input)
            amo_out = self.amo_out_module(amo_feat)
            vis_disc = self.vis_disc(vis_out)
            amo_disc = self.amo_disc(amo_out)
            domain_gt = torch.ones_like(vis_disc) if domain_is_src==True else torch.zeros_like(vis_disc)
            vis_disc_loss = F.binary_cross_entropy(vis_disc.sigmoid(), domain_gt)
            amo_disc_loss = F.binary_cross_entropy(amo_disc.sigmoid(), domain_gt)
            loss['roi_input'] = (vis_disc_loss + amo_disc_loss) * self.acs_w_roifeat

        if self.acs_w_visfeat > 0.0:
            vis_feat_gt = GradReverse.apply(vis_feat_gt)
            vis_out = self.vis_out_module(vis_feat_gt)
            amo_branch_input = torch.cat([roi_input, vis_feat_gt], dim=1)
            amo_feat = self.amo_feat_module(amo_branch_input)
            amo_out = self.amo_out_module(amo_feat)
            vis_disc = self.vis_disc(vis_out)
            amo_disc = self.amo_disc(amo_out)
            vis_disc_loss = F.binary_cross_entropy(vis_disc.sigmoid(), domain_gt)
            amo_disc_loss = F.binary_cross_entropy(amo_disc.sigmoid(), domain_gt)
            loss['vis_feat'] = (vis_disc_loss + amo_disc_loss) * self.acs_w_visfeat

        if self.acs_w_amofeat > 0.0:
            vis_out_gt = GradReverse.apply(vis_out_gt)
            vis_disc = self.vis_disc(vis_out_gt)
            vis_disc_loss = F.binary_cross_entropy(vis_disc.sigmoid(), domain_gt)
            loss['vis_out'] = vis_disc_loss * self.acs_w_amofeat

        if self.acs_w_visout > 0.0:
            amo_feat_gt = GradReverse.apply(amo_feat_gt)
            amo_out = self.amo_out_module(amo_feat_gt)
            amo_disc = self.amo_disc(amo_out)
            amo_disc_loss = F.binary_cross_entropy(amo_disc.sigmoid(), domain_gt)
            loss['amo_feat'] = amo_disc_loss * self.acs_w_visout

        if self.acs_w_amoout > 0.0:
            amo_out_gt = GradReverse.apply(amo_out_gt)
            amo_disc = self.amo_disc(amo_out_gt)
            amo_disc_loss = F.binary_cross_entropy(amo_disc.sigmoid(), domain_gt)
            loss['amo_out'] = amo_disc_loss * self.acs_w_amoout

        return loss


class _ResNetBlock(nn.Module):
    def __init__(self, chn_n, kernel_size, stride=1, padding=0, relu_is_leaky=True, add_bn=True, drop_out=0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(chn_n, chn_n, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU() if relu_is_leaky is True else nn.ReLU()
        self.bn = nn.BatchNorm2d(chn_n) if add_bn is True else (lambda x:x)
        self.drop_out = nn.Dropout2d(drop_out) if drop_out > 0 else (lambda x:x)

    def forward(self, x):
        x_init = x
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.drop_out(x)

        if x_init.shape != x.shape:
            x_init = F.adaptive_avg_pool2d(x_init, x.shape[2:])
        return x + x_init
