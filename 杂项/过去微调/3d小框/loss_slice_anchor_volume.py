import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.dual_branch.simple_loss import SparseSliceLoss
from utils.dual_cl.loss import SliceContrastiveLoss


class SliceAnchoredVolumeContrastiveLoss(nn.Module):
    """
    Slice-anchored 3D patch contrastive learning.

    Reliable vessel/background anchors are sampled from annotated slice regions. Around each anchor,
    a 3D feature cube is extracted to build EMA vessel/background volume prototypes. On unlabeled
    data, uncertain voxels are sampled as 3D cubes and compared against these slice-anchored volume
    prototypes. Only feature-confident uncertain cubes are used as positive vessel/background
    candidates, so uncertain regions are mined by similarity instead of being forced by hard pseudo
    labels.
    """

    def __init__(self, cfg):
        super().__init__()
        self.feat_dim = int(cfg.get("feat_dim", 32))
        self.cube_size = int(cfg.get("cube_size", 16))
        self.num_anchor_cubes = int(cfg.get("num_anchor_cubes", 8))
        self.num_uncertain_cubes = int(cfg.get("num_uncertain_cubes", 32))
        self.select_ratio = float(cfg.get("select_ratio", 0.25))
        self.temperature = float(cfg.get("temperature", 0.1))
        self.proto_momentum = float(cfg.get("proto_momentum", 0.95))
        self.ignore_index = int(cfg.get("ignore_index", 255))

        self.uncertain_low = float(cfg.get("uncertain_low", 0.35))
        self.uncertain_high = float(cfg.get("uncertain_high", 0.65))
        self.disagreement_thr = float(cfg.get("disagreement_thr", 0.2))
        self.min_score_margin = float(cfg.get("min_score_margin", 0.0))
        self.min_class_voxels = int(cfg.get("min_class_voxels", 1))

        self.register_buffer("vessel_vol_proto", F.normalize(torch.randn(self.feat_dim), dim=0))
        self.register_buffer("bg_vol_proto", F.normalize(torch.randn(self.feat_dim), dim=0))

    def _resize_to_feat(self, x, feat, mode="nearest"):
        if x.shape[2:] == feat.shape[2:]:
            return x
        if mode == "nearest":
            return F.interpolate(x.float(), size=feat.shape[2:], mode="nearest")
        return F.interpolate(x.float(), size=feat.shape[2:], mode="trilinear", align_corners=False)

    def _sample_coords(self, mask, max_n):
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.numel() == 0:
            return coords
        if coords.shape[0] > max_n:
            perm = torch.randperm(coords.shape[0], device=coords.device)[:max_n]
            coords = coords[perm]
        return coords

    def _cube_bounds(self, center, shape):
        d, h, w = [int(v) for v in center]
        D, H, W = shape
        s = self.cube_size
        ds = max(0, min(d - s // 2, max(0, D - s)))
        hs = max(0, min(h - s // 2, max(0, H - s)))
        ws = max(0, min(w - s // 2, max(0, W - s)))
        de, he, we = min(D, ds + s), min(H, hs + s), min(W, ws + s)
        return ds, de, hs, he, ws, we

    def _pad_cube(self, cube, value=0.0):
        s = self.cube_size
        d, h, w = cube.shape[-3:]
        pd, ph, pw = s - d, s - h, s - w
        if pd <= 0 and ph <= 0 and pw <= 0:
            return cube
        return F.pad(cube, (0, max(0, pw), 0, max(0, ph), 0, max(0, pd)), value=value)

    def _crop_cube(self, feat_b, aux_b, center, aux_pad_value=0.0):
        ds, de, hs, he, ws, we = self._cube_bounds(center, feat_b.shape[1:])
        feat_cube = feat_b[:, ds:de, hs:he, ws:we]
        aux_cube = aux_b[:, ds:de, hs:he, ws:we] if aux_b is not None else None
        feat_cube = self._pad_cube(feat_cube, value=0.0)
        if aux_cube is not None:
            aux_cube = self._pad_cube(aux_cube, value=aux_pad_value)
        return feat_cube, aux_cube

    def _weighted_token(self, feat_cube, weight_cube):
        C = feat_cube.shape[0]
        feat_flat = feat_cube.reshape(C, -1)
        weight = weight_cube.reshape(-1).float().clamp_min(0.0)
        if weight.sum() < self.min_class_voxels:
            return None
        token = torch.matmul(feat_flat, weight.unsqueeze(1)).squeeze(1) / (weight.sum() + 1e-6)
        return token

    def _mean_token(self, feat_cube):
        return feat_cube.reshape(feat_cube.shape[0], -1).mean(dim=1)

    def _collect_labeled_tokens(self, feat, mask, logits):
        mask = self._resize_to_feat(mask, feat, mode="nearest")
        prob = torch.sigmoid(self._resize_to_feat(logits, feat, mode="trilinear"))
        vessel_tokens, bg_tokens = [], []

        for b in range(feat.shape[0]):
            feat_b = feat[b]
            mask_b = mask[b]
            prob_b = prob[b]
            vessel_coords = self._sample_coords(mask_b[0] == 1, self.num_anchor_cubes)
            bg_coords = self._sample_coords(mask_b[0] == 0, self.num_anchor_cubes)

            for coord in vessel_coords.cpu().tolist():
                f_cube, p_cube = self._crop_cube(feat_b, prob_b, coord)
                token = self._weighted_token(f_cube, p_cube[0])
                if token is not None:
                    vessel_tokens.append(token)

            for coord in bg_coords.cpu().tolist():
                f_cube, p_cube = self._crop_cube(feat_b, prob_b, coord)
                token = self._weighted_token(f_cube, 1.0 - p_cube[0])
                if token is not None:
                    bg_tokens.append(token)

        v = torch.stack(vessel_tokens, dim=0) if vessel_tokens else None
        b = torch.stack(bg_tokens, dim=0) if bg_tokens else None
        return v, b

    def _collect_uncertain_tokens(self, feat, pred1, pred2):
        prob1 = torch.sigmoid(self._resize_to_feat(pred1, feat, mode="trilinear"))
        prob2 = torch.sigmoid(self._resize_to_feat(pred2, feat, mode="trilinear"))
        avg = 0.5 * (prob1 + prob2)
        diff = torch.abs(prob1 - prob2)
        uncertain = ((avg > self.uncertain_low) & (avg < self.uncertain_high)) | (diff > self.disagreement_thr)

        tokens = []
        for b in range(feat.shape[0]):
            coords = self._sample_coords(uncertain[b, 0], self.num_uncertain_cubes)
            for coord in coords.cpu().tolist():
                f_cube, _ = self._crop_cube(feat[b], None, coord)
                tokens.append(self._mean_token(f_cube))
        return torch.stack(tokens, dim=0) if tokens else None

    @torch.no_grad()
    def update_prototypes(self, vessel_tokens, bg_tokens):
        if vessel_tokens is not None and vessel_tokens.numel() > 0:
            mean = F.normalize(vessel_tokens.mean(dim=0), dim=0)
            new_proto = F.normalize(
                self.proto_momentum * self.vessel_vol_proto + (1.0 - self.proto_momentum) * mean,
                dim=0,
            )
            self.vessel_vol_proto.copy_(new_proto)

        if bg_tokens is not None and bg_tokens.numel() > 0:
            mean = F.normalize(bg_tokens.mean(dim=0), dim=0)
            new_proto = F.normalize(
                self.proto_momentum * self.bg_vol_proto + (1.0 - self.proto_momentum) * mean,
                dim=0,
            )
            self.bg_vol_proto.copy_(new_proto)

    def _prototype_nce(self, tokens, target_is_vessel):
        tokens = F.normalize(tokens, dim=1)
        sim_v = torch.matmul(tokens, self.vessel_vol_proto)
        sim_b = torch.matmul(tokens, self.bg_vol_proto)
        pos = sim_v if target_is_vessel else sim_b
        neg = sim_b if target_is_vessel else sim_v
        pos_exp = torch.exp(pos / self.temperature)
        neg_exp = torch.exp(neg / self.temperature)
        return -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)).mean()

    def _uncertain_contrast(self, uncertain_tokens):
        if uncertain_tokens is None or uncertain_tokens.numel() == 0:
            return torch.tensor(0.0, device=self.vessel_vol_proto.device)

        tokens = F.normalize(uncertain_tokens, dim=1)
        sim_v = torch.matmul(tokens, self.vessel_vol_proto)
        sim_b = torch.matmul(tokens, self.bg_vol_proto)
        score = sim_v - sim_b

        k = max(1, int(round(tokens.shape[0] * self.select_ratio)))
        vessel_idx = torch.nonzero(score > self.min_score_margin, as_tuple=False).flatten()
        bg_idx = torch.nonzero(score < -self.min_score_margin, as_tuple=False).flatten()

        losses = []
        if vessel_idx.numel() > 0:
            chosen = vessel_idx[torch.topk(score[vessel_idx], k=min(k, vessel_idx.numel())).indices]
            losses.append(self._prototype_nce(uncertain_tokens[chosen], target_is_vessel=True))
        if bg_idx.numel() > 0:
            chosen = bg_idx[torch.topk(-score[bg_idx], k=min(k, bg_idx.numel())).indices]
            losses.append(self._prototype_nce(uncertain_tokens[chosen], target_is_vessel=False))

        if not losses:
            return torch.tensor(0.0, device=uncertain_tokens.device)
        return torch.stack(losses).mean()

    def forward_labeled(self, feat, mask, logits, update_proto=True):
        vessel_tokens, bg_tokens = self._collect_labeled_tokens(feat, mask, logits)
        if update_proto:
            self.update_prototypes(
                vessel_tokens.detach() if vessel_tokens is not None else None,
                bg_tokens.detach() if bg_tokens is not None else None,
            )

        losses = []
        if vessel_tokens is not None:
            losses.append(self._prototype_nce(vessel_tokens, target_is_vessel=True))
        if bg_tokens is not None:
            losses.append(self._prototype_nce(bg_tokens, target_is_vessel=False))
        if not losses:
            return torch.tensor(0.0, device=feat.device)
        return torch.stack(losses).mean()

    def forward_unlabeled(self, feat, pred1, pred2):
        uncertain_tokens = self._collect_uncertain_tokens(feat, pred1, pred2)
        return self._uncertain_contrast(uncertain_tokens)


class DualBranchLoss(nn.Module):
    def __init__(self, base_sup_loss, pseudo_loss_fn, cl_cfg=None, volume_cl_cfg=None,
                 ramp_epochs=50, max_pseudo_weight=0.5):
        super().__init__()
        self.sup_loss_fn = base_sup_loss
        self.pseudo_loss_fn = pseudo_loss_fn
        self.ramp_epochs = ramp_epochs
        self.max_pseudo_weight = max_pseudo_weight

        self.cl_cfg = cl_cfg
        self.enable_cl = cl_cfg.get("enable", False) if cl_cfg else False
        if self.enable_cl:
            self.cl_loss_fn = SliceContrastiveLoss(cl_cfg)
            self.cl_weight = float(cl_cfg.get("weight", 0.1))
            self.cl_warmup = int(cl_cfg.get("warmup_epochs", 20))

        self.volume_cl_cfg = volume_cl_cfg
        self.enable_volume_cl = volume_cl_cfg.get("enable", False) if volume_cl_cfg else False
        if self.enable_volume_cl:
            self.volume_cl_loss_fn = SliceAnchoredVolumeContrastiveLoss(volume_cl_cfg)
            self.volume_cl_weight = float(volume_cl_cfg.get("weight", 0.1))
            self.volume_cl_warmup = int(volume_cl_cfg.get("warmup_epochs", 20))

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def _supervised_loss(self, pred, mask, img=None):
        if isinstance(self.sup_loss_fn, SparseSliceLoss):
            return self.sup_loss_fn(pred, mask, img)
        return self.sup_loss_fn(pred, mask)

    def _make_reliable_pseudo(self, pred1, pred2):
        with torch.no_grad():
            pseudo_1 = (torch.sigmoid(pred1) > 0.5).float()
            pseudo_2 = (torch.sigmoid(pred2) > 0.5).float()
        return pseudo_1, pseudo_2

    def forward(self, preds_l, mask_l, preds_u, feats_l, feats_u, current_epoch, img_l=None):
        pred1_l, pred2_l = preds_l
        pred1_u, pred2_u = preds_u
        feat1_l, feat2_l = feats_l
        feat1_u, feat2_u = feats_u

        loss_sup = 0.5 * (
            self._supervised_loss(pred1_l, mask_l, img_l)
            + self._supervised_loss(pred2_l, mask_l, img_l)
        )

        pseudo_1, pseudo_2 = self._make_reliable_pseudo(pred1_u, pred2_u)
        loss_ps = 0.5 * (
            self.pseudo_loss_fn(pred1_u, pseudo_2)
            + self.pseudo_loss_fn(pred2_u, pseudo_1)
        )

        loss_cl = torch.tensor(0.0, device=pred1_l.device)
        if self.enable_cl:
            loss_cl_l1 = self.cl_loss_fn(feat1_l, mask_l, is_gt=True, update_proto=True)
            loss_cl_l2 = self.cl_loss_fn(feat2_l, mask_l, is_gt=True, update_proto=True)
            loss_cl = 0.5 * (loss_cl_l1 + loss_cl_l2)

            if current_epoch >= self.cl_warmup:
                reliable_mask = (torch.abs(torch.sigmoid(pred1_u) - torch.sigmoid(pred2_u)) < 0.2).float()
                p_mask_1 = pseudo_1.clone()
                p_mask_1[reliable_mask == 0] = 255
                p_mask_2 = pseudo_2.clone()
                p_mask_2[reliable_mask == 0] = 255
                loss_cl_u1 = self.cl_loss_fn(feat1_u, p_mask_1, is_gt=False, update_proto=False)
                loss_cl_u2 = self.cl_loss_fn(feat2_u, p_mask_2, is_gt=False, update_proto=False)
                loss_cl = loss_cl + 0.5 * (loss_cl_u1 + loss_cl_u2)

        loss_vol = torch.tensor(0.0, device=pred1_l.device)
        if self.enable_volume_cl:
            loss_vol_l1 = self.volume_cl_loss_fn.forward_labeled(feat1_l, mask_l, pred1_l, update_proto=True)
            loss_vol_l2 = self.volume_cl_loss_fn.forward_labeled(feat2_l, mask_l, pred2_l, update_proto=True)
            loss_vol = 0.5 * (loss_vol_l1 + loss_vol_l2)

            if current_epoch >= self.volume_cl_warmup:
                loss_vol_u1 = self.volume_cl_loss_fn.forward_unlabeled(feat1_u, pred1_u, pred2_u)
                loss_vol_u2 = self.volume_cl_loss_fn.forward_unlabeled(feat2_u, pred1_u, pred2_u)
                loss_vol = loss_vol + 0.5 * (loss_vol_u1 + loss_vol_u2)

        rampup_weight = self.sigmoid_rampup(current_epoch, self.ramp_epochs)
        total_loss = loss_sup + self.max_pseudo_weight * rampup_weight * loss_ps
        if self.enable_cl:
            total_loss = total_loss + self.cl_weight * loss_cl
        if self.enable_volume_cl:
            total_loss = total_loss + self.volume_cl_weight * loss_vol

        # The existing PLModule logs the fourth returned value as loss_cl. Return the sum of both
        # contrastive terms so old logging remains valid without changing module.py.
        return total_loss, loss_sup, loss_ps, loss_cl + loss_vol
