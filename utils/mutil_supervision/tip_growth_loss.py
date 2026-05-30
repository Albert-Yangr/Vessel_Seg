import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridGrowthLoss(nn.Module):
    def __init__(self,
                 bce_weight=1.0,
                 dice_weight=1.0,
                 # 1. 静态管道亲和力 (负责变粗)
                 tube_affinity_weight=5.0,
                 # 2. 动态尖端亲和力 (负责变长 - 您的探照灯)
                 tip_affinity_weight=10.0,
                 # 3. 连通性 (负责接断)
                 connectivity_weight=1.0,

                 growth_radius=5,
                 cone_angle=40,
                 ignore_index=255,
                 **kwargs):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.tube_affinity_weight = float(tube_affinity_weight)
        self.tip_affinity_weight = float(tip_affinity_weight)
        self.connectivity_weight = float(connectivity_weight)

        self.growth_radius = int(growth_radius)
        self.ignore_index = int(ignore_index)
        self.cone_angle_rad = math.radians(cone_angle)

        # === 预计算组件 ===
        # 1. 邻居核 (找端点)
        self.neighbor_kernel = torch.ones((1, 1, 3, 3, 3), requires_grad=False)
        self.neighbor_kernel[0, 0, 1, 1, 1] = 0

        # 2. 高斯核 (平滑)
        self.blur_kernel = self._get_gaussian_kernel()

        # 3. 锥形方向核 (探照灯) - 预计算
        self.register_buffer('direction_kernels', self._generate_directional_kernels())

    # -------------------------------------------------------------------------
    # 辅助函数
    # -------------------------------------------------------------------------
    def _get_gaussian_kernel(self, kernel_size=3, sigma=0.5):
        """生成 3D 高斯核"""
        x = torch.arange(kernel_size) - kernel_size // 2
        grid = torch.stack(torch.meshgrid([x, x, x], indexing='ij'))
        kernel = torch.exp(-torch.sum(grid ** 2, dim=0) / (2 * sigma ** 2))
        return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)

    def _generate_directional_kernels(self):
        """
        预计算 26 个方向的锥形 Mask
        """
        k_size = 2 * self.growth_radius + 1
        center = self.growth_radius
        grid_range = torch.arange(k_size) - center
        Z, Y, X = torch.meshgrid(grid_range, grid_range, grid_range, indexing='ij')

        # 计算每个体素的方向向量
        voxel_vectors = torch.stack([Z, Y, X], dim=-1).float()
        voxel_norms = torch.norm(voxel_vectors, dim=-1) + 1e-6

        kernels = []
        offsets = []

        # 遍历 26 个邻域方向
        for z in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    if z == 0 and y == 0 and x == 0: continue
                    offsets.append(torch.tensor([z, y, x]).float())

        for neighbor_vec in offsets:
            # 生长方向是邻居向量的反方向
            growth_dir = -neighbor_vec
            growth_dir = growth_dir / (torch.norm(growth_dir) + 1e-6)

            # 计算余弦相似度
            cosine_sim = (voxel_vectors * growth_dir).sum(dim=-1) / voxel_norms

            # 生成锥形掩膜
            angle_mask = cosine_sim >= math.cos(self.cone_angle_rad)
            dist_mask = voxel_norms <= self.growth_radius

            kernel = (angle_mask & dist_mask).float()
            kernel[center, center, center] = 1.0  # 保证中心有值
            kernels.append(kernel)

        return torch.stack(kernels).unsqueeze(1)

    def soft_skeletonize(self, img, iter_num=3):
        """骨架提取：高斯平滑 -> 腐蚀"""
        if img.device != self.blur_kernel.device:
            self.blur_kernel = self.blur_kernel.to(img.device)
        smoothed = F.conv3d(img, self.blur_kernel, padding=1)
        skel = smoothed.clone()
        for _ in range(iter_num):
            eroded = -F.max_pool3d(-skel, 3, 1, 1)
            skel = F.relu(eroded)
        return skel

    def get_endpoints_and_directions(self, mask_binary):
        """检测端点及其朝向"""
        # 1. 提取骨架
        skeleton = self.soft_skeletonize(mask_binary.float())
        skeleton = (skeleton > 0.5).float()

        if skeleton.sum() == 0:
            return torch.zeros_like(mask_binary), None

        # 2. 计算邻居数
        num_neighbors = F.conv3d(skeleton, self.neighbor_kernel.to(skeleton.device), padding=1)

        # 3. 判定端点 (骨架点 且 邻居<=1)
        is_endpoint = (skeleton == 1) & (num_neighbors <= 1)

        if is_endpoint.sum() == 0:
            return is_endpoint.float(), None

        # 4. 计算方向
        direction_maps = []
        padded_skel = F.pad(skeleton, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        for z in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    if z == 0 and y == 0 and x == 0: continue
                    shifted = padded_skel[:, :, 1 + z: 1 + z + skeleton.shape[2],
                    1 + y: 1 + y + skeleton.shape[3],
                    1 + x: 1 + x + skeleton.shape[4]]

                    direction_map = is_endpoint & (shifted == 1)
                    direction_maps.append(direction_map)

        directions = torch.cat(direction_maps, dim=1).float()
        return is_endpoint.float(), directions

    def get_directional_growth_zone(self, directions, reference_tensor):
        """
        根据端点方向生成锥形生长区
        Args:
            directions: (B, 26, D, H, W) 或 None
            reference_tensor: 用于获取 Device 和 Shape 的参考张量 (如 target_core)
        """
        # 【修复重点】使用 reference_tensor 确保设备一致
        if directions is None or directions.sum() == 0:
            # 直接生成全零张量，设备与 target_core 一致
            return torch.zeros_like(reference_tensor)

        kernels = self.direction_kernels.to(directions.device)
        # 使用分组卷积
        growth_zone = F.conv3d(directions, kernels, padding=self.growth_radius, groups=26)

        # 合并所有方向
        growth_zone = torch.sum(growth_zone, dim=1, keepdim=True)
        return (growth_zone > 0).float()

    # -------------------------------------------------------------------------
    # 核心 Loss 计算模块
    # -------------------------------------------------------------------------

    def compute_affinity_loss(self, probs, images, mask, sigma=1.0):
        """通用亲和力计算 (用于 Tube 和 Tip)"""
        if mask.sum() == 0: return torch.tensor(0.0, device=probs.device)
        loss_sum = 0.0
        for dim in [2, 3, 4]:  # D, H, W
            img_curr = torch.narrow(images, dim, 0, images.size(dim) - 1)
            img_next = torch.narrow(images, dim, 1, images.size(dim) - 1)
            prob_curr = torch.narrow(probs, dim, 0, probs.size(dim) - 1)
            prob_next = torch.narrow(probs, dim, 1, probs.size(dim) - 1)
            mask_curr = torch.narrow(mask, dim, 0, mask.size(dim) - 1)

            # 图像越像，权重越大
            diff_img = torch.abs(img_curr - img_next)
            weight = torch.exp(-diff_img / sigma)
            diff_prob = torch.pow(prob_curr - prob_next, 2)

            loss_sum += (weight * diff_prob * mask_curr).sum() / (mask_curr.sum() + 1e-6)
        return loss_sum

    def compute_connectivity_loss(self, probs):
        """连通性约束"""
        skel = self.soft_skeletonize(probs, iter_num=3)
        if skel.sum() < 1: return torch.tensor(0.0, device=probs.device)

        num_neighbors = F.conv3d(skel, self.neighbor_kernel.to(probs.device), padding=1)
        penalty = F.relu(1.5 - num_neighbors) * skel
        return penalty.sum() / (skel.sum() + 1e-6)

    def forward(self, pred_logits, pseudo_label, images):
        """
        pseudo_label: 1=Core(Eroded), 0=BG(Dilated), 255=Tube(Uncertain)
        """
        if pseudo_label.ndim == 4: pseudo_label = pseudo_label.unsqueeze(1)

        probs = torch.sigmoid(pred_logits)
        # Target 只有 1 是核心
        target_core = (pseudo_label == 1).float()

        # ========================================
        # 1. 基础监督 (BCE + Dice)
        # ========================================
        # 这里的 valid_mask 排除 255
        valid_mask = (pseudo_label != self.ignore_index).float()

        loss_base = torch.tensor(0.0, device=probs.device)
        if valid_mask.sum() > 0:
            bce = F.binary_cross_entropy_with_logits(pred_logits, target_core, reduction='none')
            loss_bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-6)

            inter = (probs * target_core * valid_mask).sum()
            union = ((probs + target_core) * valid_mask).sum()
            loss_dice = 1.0 - (2.0 * inter) / (union + 1e-5)

            loss_base = self.bce_weight * loss_bce + self.dice_weight * loss_dice

        # ========================================
        # 2. 静态管道亲和力 (Tube Affinity)
        # ========================================
        tube_mask = (pseudo_label == self.ignore_index).float()
        loss_tube = torch.tensor(0.0, device=probs.device)

        if self.tube_affinity_weight > 0 and tube_mask.sum() > 0:
            loss_tube = self.compute_affinity_loss(probs, images, tube_mask)

        # ========================================
        # 3. 动态尖端探照灯 (Tip Growth)
        # ========================================
        loss_tip = torch.tensor(0.0, device=probs.device)

        if self.tip_affinity_weight > 0:
            with torch.no_grad():
                # 寻找核心的端点
                tips, directions = self.get_endpoints_and_directions(target_core)

                # 【修复重点】传入 target_core 作为参考，解决设备不匹配问题
                growth_zone = self.get_directional_growth_zone(directions, reference_tensor=target_core)

                # 生长区不包括已经确信的核心
                final_growth_mask = growth_zone * (1 - target_core)

            if final_growth_mask.sum() > 0:
                loss_tip = self.compute_affinity_loss(probs, images, final_growth_mask)

        # ========================================
        # 4. 连通性约束
        # ========================================
        loss_conn = torch.tensor(0.0, device=probs.device)
        if self.connectivity_weight > 0:
            loss_conn = self.compute_connectivity_loss(probs)

        # 汇总
        return loss_base + \
            (self.tube_affinity_weight * loss_tube) + \
            (self.tip_affinity_weight * loss_tip) + \
            (self.connectivity_weight * loss_conn)