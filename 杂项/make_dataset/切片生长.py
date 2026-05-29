import os

# 【核心修改】：强制让 PyTorch 只能看到 2 号 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import time


def evaluate_proxy_accuracy(proxy_label_path, gt_label_path, sparse_label_path):
    """
    究极评估模块：同时包含【血管与背景双向评估】以及【与原始切片的增益对比】
    """
    print("\n=================== 🌟 代理标签增益与质量体检报告 🌟 ===================")

    if not os.path.exists(gt_label_path) or not os.path.exists(sparse_label_path):
        print("⚠️ 找不到真实标签或原始切片文件，无法执行全量对比评估。")
        return

    # 1. 加载三种状态的标签
    proxy_nii = nib.load(proxy_label_path)
    proxy_data = proxy_nii.get_fdata().astype(np.uint8)

    gt_nii = nib.load(gt_label_path)
    gt_data = gt_nii.get_fdata().astype(np.uint8)

    sparse_nii = nib.load(sparse_label_path)
    sparse_data = sparse_nii.get_fdata().astype(np.uint8)

    if proxy_data.shape != gt_data.shape:
        print("⚠️ 代理标签与真实标签尺寸不一致，无法评估！")
        return

    # 2. 提取掩码
    proxy_vessel = (proxy_data == 1)
    proxy_bg = (proxy_data == 0)
    proxy_unknown = (proxy_data == 255)

    gt_vessel = (gt_data == 1)
    gt_bg = (gt_data == 0)

    sparse_vessel = (sparse_data == 1)
    sparse_bg = (sparse_data == 0)

    # 3. 基础体积统计
    total_voxels = proxy_data.size
    gt_vessel_count = np.sum(gt_vessel)
    gt_bg_count = np.sum(gt_bg)

    # 4. 原始切片(Sparse)基线性能
    correct_sparse_vessel = np.sum(sparse_vessel & gt_vessel)
    sparse_vessel_recall = correct_sparse_vessel / (gt_vessel_count + 1e-8) * 100
    sparse_vessel_precision = correct_sparse_vessel / (np.sum(sparse_vessel) + 1e-8) * 100

    correct_sparse_bg = np.sum(sparse_bg & gt_bg)
    sparse_bg_recall = correct_sparse_bg / (gt_bg_count + 1e-8) * 100
    sparse_bg_precision = correct_sparse_bg / (np.sum(sparse_bg) + 1e-8) * 100

    # 5. 代理标签(Proxy)最终性能
    correct_proxy_vessel = np.sum(proxy_vessel & gt_vessel)
    proxy_vessel_recall = correct_proxy_vessel / (gt_vessel_count + 1e-8) * 100
    proxy_vessel_precision = correct_proxy_vessel / (np.sum(proxy_vessel) + 1e-8) * 100

    err_bg_as_vessel = np.sum(proxy_vessel & gt_bg)  # 血管混入了背景 (FP)
    err_vessel_as_bg = np.sum(proxy_bg & gt_vessel)  # 背景混入了血管 (FN)
    ignored_vessel = np.sum(proxy_unknown & gt_vessel)

    correct_proxy_bg = np.sum(proxy_bg & gt_bg)
    proxy_bg_recall = correct_proxy_bg / (gt_bg_count + 1e-8) * 100
    proxy_bg_precision = correct_proxy_bg / (np.sum(proxy_bg) + 1e-8) * 100
    ignored_bg = np.sum(proxy_unknown & gt_bg)

    # ================= 打印究极战报 =================
    print(f"【宏观体积大考】 总像素: {total_voxels}")
    print(f"  ▶ 血管体积: 原始切片 {np.sum(sparse_vessel):>8} -> 算法扩充至 {np.sum(proxy_vessel):>8} 体素")
    print(f"  ▶ 背景体积: 原始切片 {np.sum(sparse_bg):>8} -> 算法扩充至 {np.sum(proxy_bg):>8} 体素")
    print(f"  ▶ 未知区域: 算法保留了 {np.sum(proxy_unknown):>8} 体素作为安全地带 (255)\n")

    print(f"【🔴 血管专项评估 (Vessel)】 真实血管总体积: {gt_vessel_count}")
    print(f"  🏆 [体积召回率 Recall] —— 我们挖出了多少隐藏血管？")
    print(f"      切片基线: {sparse_vessel_recall:>6.2f}%")
    print(
        f"      算法扩充: {proxy_vessel_recall:>6.2f}%  (🚀 净增长: +{(proxy_vessel_recall - sparse_vessel_recall):.2f}%)")

    print(f"  🛡️ [质量纯度 Precision] —— 扩出来的血管干净吗？")
    print(f"      切片基线: {sparse_vessel_precision:>6.2f}%")
    print(
        f"      算法扩充: {proxy_vessel_precision:>6.2f}%  (📉 质量损耗: {(proxy_vessel_precision - sparse_vessel_precision):.2f}%, 混入伪影 {err_bg_as_vessel} 个)")

    print(f"  ⏸️ 安全丢弃 (设为255): {ignored_vessel} 个 ({ignored_vessel / gt_vessel_count * 100:.2f}%)")
    print(
        f"  ❌ 致命误伤 (标为背景): {err_vessel_as_bg} 个 ({err_vessel_as_bg / gt_vessel_count * 100:.2f}%) <- 此项过高说明背景扩散太猛\n")

    print(f"【🔵 背景专项评估 (Background)】 真实背景总体积: {gt_bg_count}")
    print(f"  🏆 [体积召回率 Recall] —— 我们铺开了多少安全区？")
    print(f"      切片基线: {sparse_bg_recall:>6.2f}%")
    print(f"      算法扩充: {proxy_bg_recall:>6.2f}%  (🚀 净增长: +{(proxy_bg_recall - sparse_bg_recall):.2f}%)")

    print(f"  🛡️ [质量纯度 Precision] —— 安全区里混进血管了吗？")
    print(f"      切片基线: {sparse_bg_precision:>6.2f}%")
    print(
        f"      算法扩充: {proxy_bg_precision:>6.2f}%  (📉 质量损耗: {(proxy_bg_precision - sparse_bg_precision):.2f}%, 吞噬血管 {err_vessel_as_bg} 个)")
    print(f"  ⏸️ 安全丢弃 (设为255): {ignored_bg} 个 ({ignored_bg / gt_bg_count * 100:.2f}%)")
    print("====================================================================\n")


def pytorch_asymmetric_random_walk(img_path, label_path, output_path,
                                   beta_vessel=150, iter_vessel=300,
                                   beta_bg=500, iter_bg=100,
                                   conflict_threshold=0.3):
    """
    非对称双轨版 3D 随机游走 (完整精度满血版)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f"1. 环境初始化完毕，当前挂载硬件: {device} (物理 GPU)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("2. 加载数据与预处理...")
    img_nii = nib.load(img_path)
    img_data = img_nii.get_fdata()
    label_nii = nib.load(label_path)
    label_data = label_nii.get_fdata()

    img_min, img_max = np.min(img_data), np.max(img_data)
    if img_max > img_min:
        img_data = (img_data - img_min) / (img_max - img_min)

    img_t = torch.tensor(img_data, dtype=torch.float32, device=device)
    label_t = torch.tensor(label_data, dtype=torch.float32, device=device)

    P_fg = (label_t == 1).float()
    P_bg = (label_t == 0).float()

    print(" -> 构建安全缓冲带 (挖除贴边背景)...")
    P_fg_unsqueeze = P_fg.unsqueeze(0).unsqueeze(0)
    dilated_vessel_init = F.max_pool3d(P_fg_unsqueeze, kernel_size=5, stride=1, padding=2).squeeze()
    P_bg = P_bg * (1.0 - dilated_vessel_init)

    is_seed = P_fg + P_bg
    orig_P_fg = P_fg.clone()
    orig_P_bg = P_bg.clone()

    print(f"3. 构建非对称阻力网络...")
    print(f"   - 血管阻力 beta: {beta_vessel}")
    print(f"   - 背景阻力 beta: {beta_bg}")

    diff_x_p = (img_t - torch.roll(img_t, shifts=-1, dims=0)) ** 2
    diff_x_m = (img_t - torch.roll(img_t, shifts=1, dims=0)) ** 2
    diff_y_p = (img_t - torch.roll(img_t, shifts=-1, dims=1)) ** 2
    diff_y_m = (img_t - torch.roll(img_t, shifts=1, dims=1)) ** 2
    diff_z_p = (img_t - torch.roll(img_t, shifts=-1, dims=2)) ** 2
    diff_z_m = (img_t - torch.roll(img_t, shifts=1, dims=2)) ** 2

    W_v_xp, W_b_xp = torch.exp(-beta_vessel * diff_x_p), torch.exp(-beta_bg * diff_x_p)
    W_v_xm, W_b_xm = torch.exp(-beta_vessel * diff_x_m), torch.exp(-beta_bg * diff_x_m)
    W_v_yp, W_b_yp = torch.exp(-beta_vessel * diff_y_p), torch.exp(-beta_bg * diff_y_p)
    W_v_ym, W_b_ym = torch.exp(-beta_vessel * diff_y_m), torch.exp(-beta_bg * diff_y_m)
    W_v_zp, W_b_zp = torch.exp(-beta_vessel * diff_z_p), torch.exp(-beta_bg * diff_z_p)
    W_v_zm, W_b_zm = torch.exp(-beta_vessel * diff_z_m), torch.exp(-beta_bg * diff_z_m)

    W_v_sum = W_v_xp + W_v_xm + W_v_yp + W_v_ym + W_v_zp + W_v_zm
    W_b_sum = W_b_xp + W_b_xm + W_b_yp + W_b_ym + W_b_zp + W_b_zm

    max_iter = max(iter_vessel, iter_bg)
    print(f"4. 开始非对称迭代推演 (总计 {max_iter} 次)...")
    start_time = time.time()

    for i in range(max_iter):
        if i < iter_vessel:
            P_fg_new = (
                               torch.roll(P_fg, shifts=1, dims=0) * W_v_xp +
                               torch.roll(P_fg, shifts=-1, dims=0) * W_v_xm +
                               torch.roll(P_fg, shifts=1, dims=1) * W_v_yp +
                               torch.roll(P_fg, shifts=-1, dims=1) * W_v_ym +
                               torch.roll(P_fg, shifts=1, dims=2) * W_v_zp +
                               torch.roll(P_fg, shifts=-1, dims=2) * W_v_zm
                       ) / W_v_sum
            P_fg = torch.where(is_seed == 1, orig_P_fg, P_fg_new)

        if i < iter_bg:
            P_bg_new = (
                               torch.roll(P_bg, shifts=1, dims=0) * W_b_xp +
                               torch.roll(P_bg, shifts=-1, dims=0) * W_b_xm +
                               torch.roll(P_bg, shifts=1, dims=1) * W_b_yp +
                               torch.roll(P_bg, shifts=-1, dims=1) * W_b_ym +
                               torch.roll(P_bg, shifts=1, dims=2) * W_b_zp +
                               torch.roll(P_bg, shifts=-1, dims=2) * W_b_zm
                       ) / W_b_sum
            P_bg = torch.where(is_seed == 1, orig_P_bg, P_bg_new)

        if (i + 1) % 50 == 0:
            print(f"   -> 已完成 {i + 1}/{max_iter} 次...")

    print(f"   推演完成！GPU 耗时: {time.time() - start_time:.2f} 秒")

    print(f"5. 生成标签并执行【三重提纯】...")
    final_label_t = torch.full_like(label_t, 255)

    final_label_t[P_bg > P_fg] = 0
    final_label_t[P_fg > P_bg] = 1

    # 【提纯 1】：消除冲突区
    conflict_mask = torch.abs(P_fg - P_bg) < conflict_threshold
    conflict_mask = conflict_mask & (is_seed == 0)
    final_label_t[conflict_mask] = 255

    # 【提纯 2】：绝对亮度守卫
    real_vessel_intensities = img_t[label_t == 1]
    if len(real_vessel_intensities) > 0:
        mean_intensity = real_vessel_intensities.mean()
        std_intensity = real_vessel_intensities.std()
        intensity_lower_bound = mean_intensity - 1.5 * std_intensity
        dark_leaks = (final_label_t == 1) & (img_t < intensity_lower_bound) & (is_seed == 0)
        final_label_t[dark_leaks] = 255
        print(f"   [亮度守卫] 拦截底线: {intensity_lower_bound:.4f}，切除伪影: {dark_leaks.sum().item()} 个")

    # 【提纯 3】：边界保护
    current_vessel = (final_label_t == 1).float().unsqueeze(0).unsqueeze(0)
    dilated_vessel_mask = F.max_pool3d(current_vessel, kernel_size=3, stride=1, padding=1).squeeze()
    boundary_conflict = (dilated_vessel_mask == 1) & (final_label_t == 0) & (is_seed == 0)
    final_label_t[boundary_conflict] = 255
    print(f"   [边界保护] 退回可疑边界: {boundary_conflict.sum().item()} 个")

    final_label = final_label_t.cpu().numpy().astype(np.uint8)

    out_nii = nib.Nifti1Image(final_label, img_nii.affine)
    nib.save(out_nii, output_path)
    print(f"代理标签已保存至: {output_path}")
    print(f"==================================================")


if __name__ == "__main__":
    root = "/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-refile/train/1/1"

    INPUT_IMAGE = root + ".img.nii.gz"
    INPUT_SPARSE_LABEL = root + ".slice.nii.gz"
    OUTPUT_PROXY_LABEL = root + ".agent.nii.gz"
    INPUT_GT_LABEL = root + ".label.nii.gz"

    # ================= 黄金调参区 =================
    #肺部600，200，0.5
    BETA_VESSEL = 1500
    ITER_VESSEL = 400

    BETA_BG = 100
    ITER_BG = 400

    THRESHOLD = 0.1
    # ==============================================

    pytorch_asymmetric_random_walk(
        img_path=INPUT_IMAGE,
        label_path=INPUT_SPARSE_LABEL,
        output_path=OUTPUT_PROXY_LABEL,
        beta_vessel=BETA_VESSEL,
        iter_vessel=ITER_VESSEL,
        beta_bg=BETA_BG,
        iter_bg=ITER_BG,
        conflict_threshold=THRESHOLD
    )

    # 重点：同时传入三个路径，生成完整的对比增益战报
    evaluate_proxy_accuracy(
        proxy_label_path=OUTPUT_PROXY_LABEL,
        gt_label_path=INPUT_GT_LABEL,
        sparse_label_path=INPUT_SPARSE_LABEL
    )