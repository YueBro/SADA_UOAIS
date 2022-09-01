import time

import numpy as np
import torch

from detectron2.engine.defaults import TrainerBase
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage

from typing import Mapping

# from IPython import embed


__all__ = [
    "DASubTrainer",
]


class DASubTrainer(TrainerBase):

    def __init__(
        self, 
        
        cfg,

        model, model_disc,

        data_loader, data_loader_sup, 
        
        optimizer, opt_disc,
    ):
        super().__init__()

        model.train()

        self.model = model
        self.model_disc = model_disc

        self.optimizer = optimizer
        self.opt_disc = opt_disc

        self.data_loader = data_loader
        self.data_loader_sup = data_loader_sup

        self._data_loader_iter_obj = iter(self.data_loader)
        self._data_loader_sup_iter_obj = iter(self.data_loader_sup)

        self.da_accuse_weight = cfg.DOMAIN_ADAPTATION.DA_ACCUSE_WEIGHT
        self.stu_disc_weight = cfg.DOMAIN_ADAPTATION.STU_DISC_WEIGHT
        self.fg_batch_max = cfg.DOMAIN_ADAPTATION.FG_BATCH_MAX

        self.cfg = cfg

    def run_step(self):

        assert self.model.training, "[DA_SubTrainer] model was changed to eval mode!"
        assert self.model_disc.training, "[DA_SubTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        # Get batched data ################################################
        data = next(self._data_loader_iter_obj)
        data_tgt = next(self._data_loader_sup_iter_obj)
        data_time = time.perf_counter() - start

        # Main Forwards ###################################################
        loss_dict_tgt, inner_bin = self.model(data_tgt, proposal_append_gt=False)
        # fore_feat_tgt = inner_bin['fore_feature']        # [B,256,w,h]*5
        # box_feat_tgt = inner_bin['box_feature']     # [NN,256,7,7]
        roi_feat_fg_tgt = inner_bin['roi_feature_fg']
        post_feat_tgt = inner_bin['post_feature']      # [N,256,14,14]*3
        output_logits_tgt = inner_bin['output_logits']      # keys=[visible, amodal], each with shape [N,1,28,28] (not sigmoid!)
        # embed()
        # print("!!!!!!! Remember to comment out embed() !!!!!!!")
        # exit(0)
        del inner_bin
        # box_feat_tgt = self.sample_feature_map(box_feat_tgt, n=self.da_box_feat_batch_max, regenerate_idxs=True)
        roi_feat_fg_tgt = self.sample_feature_map(roi_feat_fg_tgt, n=self.fg_batch_max, regenerate_idxs=True)
        post_feat_tgt = self.sample_feature_map(post_feat_tgt, n=self.fg_batch_max, regenerate_idxs=False)
        output_logits_tgt = self.sample_feature_map(output_logits_tgt, n=self.fg_batch_max, regenerate_idxs=False)
        del data_tgt
        loss_tgt = sum(loss_dict_tgt.values()).detach().cpu()  # Only for inference
        
        loss_dict_src, inner_bin = self.model(data)
        # fore_feat_src = inner_bin['fore_feature']
        # box_feat_src = inner_bin['box_feature']
        roi_feat_fg_src = inner_bin['roi_feature_fg']
        post_feat_src = inner_bin['post_feature']
        output_logits_src = inner_bin['output_logits']       # keys=[visible, amodal], each with shape [N,1,28,28]
        del inner_bin
        # box_feat_src = self.sample_feature_map(box_feat_src, n=self.da_box_feat_batch_max, regenerate_idxs=True)
        roi_feat_fg_src = self.sample_feature_map(roi_feat_fg_src, n=self.fg_batch_max, regenerate_idxs=True)
        post_feat_src = self.sample_feature_map(post_feat_src, n=self.fg_batch_max, regenerate_idxs=False)
        output_logits_src = self.sample_feature_map(output_logits_src, n=self.fg_batch_max, regenerate_idxs=False)
        del data
        loss_src = sum(loss_dict_src.values())

        # Student Learn ###################################################
        stu_lrn_loss_src = self.model_disc(
            roi_feat_fg_src.detach(),
            post_feat_src[0].detach(),
            output_logits_src['visible'].detach(),
            post_feat_src[1].detach(),
            output_logits_src['amodal'].detach(),
            domain_is_src=True,
            operation='learn_and_judge',)
        stu_lrn_loss_tgt = self.model_disc(
            roi_feat_fg_tgt.detach(),
            post_feat_tgt[0].detach(),
            output_logits_tgt['visible'].detach(),
            post_feat_tgt[1].detach(),
            output_logits_tgt['amodal'].detach(),
            domain_is_src=False,
            operation='learn_and_judge',)
        stu_lrn_loss = (stu_lrn_loss_src['stuLrn_loss'] + stu_lrn_loss_tgt['stuLrn_loss']) / 2
        stu_disc_loss = (stu_lrn_loss_src['stuDisc_loss'] + stu_lrn_loss_tgt['stuDisc_loss']) / 2
        stu_loss = stu_lrn_loss + stu_disc_loss * self.stu_disc_weight

        if isinstance(stu_loss, torch.Tensor):
            self.opt_disc.zero_grad()
            stu_loss.backward()
            self.opt_disc.step()

        # DA Loss / Accuse ################################################
        for param in self.model_disc.parameters():  # Freeze student (save memory)
            param.requires_grad = False             # Freeze student (save memory)
        
        stu_acs_loss_src = self.model_disc(
            roi_feat_fg_src,
            post_feat_src[0],
            output_logits_src['visible'],
            post_feat_src[1],
            output_logits_src['amodal'],
            domain_is_src=True,
            operation='accuse',)
        stu_acs_loss_tgt = self.model_disc(
            roi_feat_fg_tgt,
            post_feat_tgt[0],
            output_logits_tgt['visible'],
            post_feat_tgt[1],
            output_logits_tgt['amodal'],
            domain_is_src=False,
            operation='accuse',)
        acs_loss = stu_acs_loss_src['stuAcs_loss'] + stu_acs_loss_tgt['stuAcs_loss']
        loss = loss_src + acs_loss * self.da_accuse_weight

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for param in self.model_disc.parameters():  # Defroze student (save memory)
            param.requires_grad = True              # Defroze student (save memory)

        # Verbose #########################################################
        loss_dict = {
            "total_loss": loss,
            "src_loss": loss_src,
            "tgt_loss": loss_tgt,
            "StuLrn_ls": stu_lrn_loss,
            "StuDis_ls": stu_disc_loss,
            "StuAcs_ls": acs_loss,
        }
        loss_dict.update({'tgt_'+k:v for k,v in loss_dict_tgt.items()})
        self._write_metrics(loss_dict, data_time)

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
    ) -> None:
        DASubTrainer.write_metrics(loss_dict, data_time)
    
    @staticmethod
    def generate_random_idxs(size, sample_size):
        idxs = np.arange(size)
        np.random.shuffle(idxs)
        idxs = np.sort(idxs[:sample_size])
        return idxs

    def sample_feature_map(self, feature_maps, n, regenerate_idxs=True):
        """
        If input is list or dict, each element will be sampled individually.
        Otherwise, sample directly on the input.
        """
        dict_keys = None
        is_list = True
        if isinstance(feature_maps, dict):
            dict_keys = feature_maps.keys()
            feature_maps = list(feature_maps.values())
        elif isinstance(feature_maps, list) is False:
            is_list = False
            feature_maps = [feature_maps]
        
        batch_n = feature_maps[0].shape[0]
        for f in feature_maps[1:]:
            assert f.shape[0]==batch_n, "Input feature maps must have same batch size"
        
        if n < batch_n:
            if regenerate_idxs is True:
                idxs = DASubTrainer.generate_random_idxs(batch_n, n)
                self._sampling_idxs = idxs
            else:
                assert hasattr(self, '_sampling_idxs'), "idxs haven't been generated yet. Unable to use \"regenerate_idxs=True\""
                idxs = self._sampling_idxs
                assert len(idxs) == n, "len(idxs)==n don't match"

            feature_maps_out = []
            for f in feature_maps:
                f = f[idxs, ...]
                feature_maps_out.append(f)
        else:
            feature_maps_out = list(feature_maps)

        if dict_keys is not None:
            feature_maps_out = {k:v for k,v in zip(dict_keys, feature_maps_out)}
        if is_list is False:
            feature_maps_out = feature_maps_out[0]
        return feature_maps_out
    
    def sample_proposals(self, proposals, n, regenerate_idxs=True):
        gt_vis_masks = []
        gt_amo_masks = []
        for instances_per_image in proposals:
            if len(instances_per_image) == 0:
                continue

            gt_masks_per_image = instances_per_image.get('gt_visible_masks').crop_and_resize(
                instances_per_image.proposal_boxes.tensor, 28
            )#.to(device=proposals.device)
            gt_vis_masks.append(gt_masks_per_image)

            gt_masks_per_image = instances_per_image.get('gt_masks').crop_and_resize(
                instances_per_image.proposal_boxes.tensor, 28
            )#.to(device=proposals.device)
            gt_amo_masks.append(gt_masks_per_image)

        if gt_vis_masks==[]:
            return {'gt_vis_masks': [], 'gt_amo_masks': []}
        gt_vis_masks = torch.cat(gt_vis_masks, dim=0)
        gt_amo_masks = torch.cat(gt_amo_masks, dim=0)
        assert gt_vis_masks.shape[0] == gt_amo_masks.shape[0]
        batch_n = gt_vis_masks.shape[0]

        if n < batch_n:
            if regenerate_idxs is True:
                idxs = DASubTrainer.generate_random_idxs(batch_n, n)
                self._sampling_idxs = idxs
            else:
                assert hasattr(self, '_sampling_idxs'), "idxs haven't been generated yet. Unable to use \"regenerate_idxs=True\""
                idxs = self._sampling_idxs
                assert len(idxs) == n, "len(idxs)==n don't match"
            
            gt_vis_masks = gt_vis_masks[idxs, ...]
            gt_amo_masks = gt_amo_masks[idxs, ...]
        
        return {'gt_vis_masks': gt_vis_masks, 'gt_amo_masks': gt_amo_masks}
        
    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
    ) -> None:
        metrics_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            metrics_dict[k] = v
        metrics_dict["data_time"] = data_time

        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = metrics_dict["total_loss"] if "total_loss" in metrics_dict else sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )
            
            # storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])
