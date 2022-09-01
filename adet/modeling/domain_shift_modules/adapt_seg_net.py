import torch
from torch import nn
import torch.nn.functional as F
from .grad_modules import GradReverse
# from adet.modeling.rcnn.mask_heads import mask_rcnn_loss


__all__ = [
    'AdaptSegNetModule',
    'AdaptSegNetModule_DoubleMasks'
]


class AdaptSegNetModule(nn.Module):
    def __init__(self, input_type: str):
        super().__init__()
        input_types = ['feature_map', 'output_mask']
        assert input_type in input_types, f"\"input_type\" must be {input_types}."
        self.input_type = input_type

        if input_type == 'feature_map':
            self.predictor = self.build_light_weight_predictor()
        else:
            self.predictor = _EmptyModule()
        
        self.discriminator = self.build_discriminator()

    @staticmethod
    def build_light_weight_predictor():
        # Input shape: [N, 256, 14, 14]
        return nn.Sequential(
            nn.ConvTranspose2d(256*3, 128, kernel_size=2, stride=2),    # [N,128,28,28]
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),               # [N,128,28,28]
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1),                            # [N,1,28,28]
        )

    @staticmethod
    def build_discriminator():
        # Input shape: [N, 1, 28, 28]
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
            nn.Sigmoid(),
        )

    def forward(self, x, label: bool, gt_masks=None):
        """
        If input_type is "feature":
            If feeding source features, label should be True, gt_masks should be provided.
            If feating target features, label should be False, gt_masks shoudl not be provided.
        If input_type is "output_mask":
            Only x and label should be provided.
        """
        assert isinstance(label, bool)
        
        # x: Feature map
        if (self.input_type == 'feature_map'):
            if x[0].shape[0] == 0:
                return 0.0
            
            x = torch.cat(x, dim=1)
            x = self.predictor(x)
            if label is True:
                assert gt_masks is not None, "With label=True (src domain), gt_masks must be provided instead of \"None\""
                prediction_loss = _mask_loss(x, gt_masks)
            else:
                prediction_loss = 0.0
        else:
            prediction_loss = 0.0

        # x: Output Mask
        if x.shape[0] == 0:
            return 0.0
        assert x.shape[1:] == torch.Size([1,28,28]), f"Output mask shape received: {tuple(x.shape)}"
        x = GradReverse.apply(x)

        logits = self.discriminator(x)
        if label is True:
            gt = torch.ones_like(logits)
        else:
            gt = torch.zeros_like(logits)
        adversarial_loss = F.binary_cross_entropy(logits, gt)

        return prediction_loss + adversarial_loss


class AdaptSegNetModule_DoubleMasks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vis_model = AdaptSegNetModule(input_type='output_mask')
        self.amo_model = AdaptSegNetModule(input_type='output_mask')
    
    def forward(self, vis_mask, amo_mask, label):
        l1 = self.vis_model(vis_mask, label)
        l2 = self.amo_model(amo_mask, label)
        return l1 + l2


class _EmptyModule(nn.Module):
    def forward(self, x):
        return x


def _mask_loss(pred_mask_logits: torch.Tensor, gt_masks):

    cls_agnostic_mask = True
    total_num_masks = pred_mask_logits.size(0)

    if len(gt_masks) == 0:
        # return pred_mask_logits.sum() * 0
        return torch.tensor(0.0).to(gt_masks.device)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = torch.cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    gt_masks = gt_masks.to(dtype=torch.float32)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss
