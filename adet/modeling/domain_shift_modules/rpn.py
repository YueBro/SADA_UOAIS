import torch

from detectron2.modeling.proposal_generator.rpn import (
    StandardRPNHead,
    RPN,
    RPN_HEAD_REGISTRY,
)
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from typing import List, Dict, Optional
from detectron2.structures import Boxes, ImageList, Instances

from detectron2.utils.registry import Registry


__all__ = [
    'StandardRPNHead_FeatureOutput',
    'RPN_FeatureOutput',
]


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead_FeatureOutput(StandardRPNHead):

    def forward(self, features: List[torch.Tensor]):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        inner_features = []
        for x in features:
            t = self.conv(x)
            inner_features.append(t)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas, inner_features


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN_FeatureOutput(RPN):
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas, inner_features = self.rpn_head(features)
        pred_objectness_logits = [
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )

        if self.training is True:
            return proposals, losses, inner_features
        else:
            return proposals, losses
