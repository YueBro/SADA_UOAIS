import torch

from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage

from typing import List, Dict


__all__ = [
    "GeneralizedRCNN_FeatureOutput"
]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_FeatureOutput(GeneralizedRCNN):
    """
    This is a modified version of GeneralizedRCNN, which outputs feature maps of backbone.
    
    Author: Yulin Shen
    """

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], proposal_append_gt=True):

        if not self.training:
            return self.inference(batched_inputs)
        else:
            inner_bin = {}

            images = self.preprocess_image(batched_inputs)
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            backbone_features = self.backbone(images.tensor)
            # inner_bin['fore_feature'] = list(backbone_features.values())

            if self.proposal_generator is not None:
                proposals, proposal_losses, rpn_features = self.proposal_generator(images, backbone_features, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}
            inner_bin['fore_feature'] = rpn_features
            
            _temp_value = self.roi_heads.proposal_append_gt
            if proposal_append_gt is False:      # for target domain
                self.roi_heads.proposal_append_gt = False
            _, detector_losses, _inner_bin = self.roi_heads(images, backbone_features, proposals, gt_instances);  inner_bin.update(_inner_bin)
            self.roi_heads.proposal_append_gt = _temp_value

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            # inner_bin keys:
            #   'fore_feature'
            #   'roi_feature'
            #   'post_feature'
            #   'box_feature'
            #   'output_logits'
            #   'proposals'

            return losses, inner_bin
