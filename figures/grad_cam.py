import os
import glob

import torch
import torch.nn.functional as F
import numpy as np
from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
import pandas as pd

from resnet_select import resnet18


def _extract_state_dict(ckpt):
    """兼容不同 checkpoint 格式"""
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "network"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_module_prefix(state_dict):
    """去掉 DataParallel 的 'module.' 前缀"""
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        out[k[7:]] = v if k.startswith("module.") else v
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def load_weights(model, weight_path, device):
    # ckpt = torch.load(weight_path, map_location=device)
    # sd = _extract_state_dict(ckpt)
    # sd = _strip_module_prefix(sd)
    # missing, unexpected = model.load_state_dict(sd, strict=False)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    # if missing or unexpected:
    #     print(f"[WARN] {os.path.basename(weight_path)} loaded with strict=False")
    #     if missing:
    #         print(f"  missing keys (up to 10): {missing[:10]}")
    #     if unexpected:
    #         print(f"  unexpected keys (up to 10): {unexpected[:10]}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = r"D:\DJC\Ablation\ADNI_data_notest_age\new_test_split.csv"
    data_img = pd.read_csv(csv_path)

    class_map = {
        0: "AD",
        1: "MCI",
        2: "NC"
    }
    # data_img_path = data_img['name']

    # ====== 你需要改的地方：权重目录 & 输出目录 ======
    weights_dir = r"D:\DJC\dual\weights"  # 存放很多 .pth 的目录
    out_root = r"D:\DJC\dual\grad_cam23"  # 输出根目录
    # ============================================

    # 收集所有权重文件
    weight_files = []
    for ext in ("*.pth", "*.pt", "*.bin"):
        weight_files += glob.glob(os.path.join(weights_dir, ext))
    weight_files = sorted(weight_files)

    if not weight_files:
        raise FileNotFoundError(f"No weight files found in: {weights_dir}")

    print(f"Using device: {device}")
    print(f"Found {len(weight_files)} weight files to process.")

    # 初始化模型（结构固定，权重循环加载）
    model = resnet18()
    model.to(device)
    model.eval()

    target_layer = [model.layer3[-1], model.layer4[-1]]
    # target_layer = [model.layer2[-1], model.layer4[-1]]
    # 固定 slice index（保持你的原逻辑）
    slice_index_1 = 105
    slice_index_2 = 86
    slice_index_3 = 50
    slice_index_4 = 50

    # 逐个权重处理
    for weight_path in weight_files:
        wname = os.path.splitext(os.path.basename(weight_path))[0]
        print(f"\n==== Processing weight: {wname} ====")

        # 加载权重
        load_weights(model, weight_path, device)

        # 每加载一个权重，重新创建 GradCAM（更稳妥）
        grad_cam = GradCAM(model=model, target_layer=target_layer)

        # 为该权重创建输出子目录，避免覆盖
        out_dir = os.path.join(out_root, wname)
        os.makedirs(out_dir, exist_ok=True)

        # 进行前向传播（对该权重下的所有图片）
        # for img_ in data_img_path:
        for _, row in data_img.iterrows():
            img_ = row['name']
            gt_class = int(row['class'])  # csv中的真实类别
            class_name = class_map.get(gt_class, f"class_{gt_class}")
            class_out_dir = os.path.join(out_dir, class_name)
            os.makedirs(class_out_dir, exist_ok=True)
            img_path1 = img_[31:-18]
            img_path2 = img_[-11:]
            img_path = r"D:\DJC\dataset/" + img_path1 + img_path2

            img = np.load(img_path)
            assert img is not None
            img = img.astype(np.float32)

            input_tensor = torch.from_numpy(img).to(device)
            input_tensor = input_tensor.unsqueeze(0)  # [1, 1, D, H, W] 假设你的 resnet18 接收该格式

            output, _ = model(input_tensor)

            # 获取预测类别
            pred_classes = torch.max(output, dim=1)[1]

            # 计算 Grad-CAM 激活图
            activation_map = grad_cam(class_idx=pred_classes.item(), scores=output)

            activation_map_resized = F.interpolate(
                activation_map[0].unsqueeze(0),
                size=(160, 160, 160),
                mode='trilinear',
                align_corners=False
            )

            # 提取单个切片后可视化
            slice_map_1 = activation_map_resized[0, 0, slice_index_1, :, :].detach().cpu().numpy()
            slice_map_2 = activation_map_resized[0, 0, :, slice_index_2, :].detach().cpu().numpy()
            slice_map_3 = activation_map_resized[0, 0, :, :, slice_index_3].detach().cpu().numpy()
            slice_map_4 = activation_map_resized[0, 0, :, slice_index_4, :].detach().cpu().numpy()

            # 原始切片
            original_slice_1 = input_tensor[0, 0, slice_index_1, :, :].detach().cpu().numpy()
            original_slice_2 = input_tensor[0, 0, :, slice_index_2, :].detach().cpu().numpy()
            original_slice_3 = input_tensor[0, 0, :, :, slice_index_3].detach().cpu().numpy()
            original_slice_4 = input_tensor[0, 0, :, slice_index_4, :].detach().cpu().numpy()

            # 画图
            fig, axs = plt.subplots(1, 4, figsize=(15, 5))

            rotated_original_slice_1 = np.rot90(original_slice_1, k=1)
            rotated_slice_map_1 = np.rot90(slice_map_1, k=1)
            axs[0].imshow(rotated_original_slice_1, cmap='gray', alpha=0.7)
            axs[0].imshow(rotated_slice_map_1, cmap='jet', alpha=0.5)
            axs[0].axis('off')
            axs[0].set_title(str(slice_index_1))

            rotated_original_slice_4 = np.rot90(original_slice_4, k=1)
            rotated_slice_map_4 = np.rot90(slice_map_4, k=1)
            axs[1].imshow(rotated_original_slice_4, cmap='gray', alpha=0.7)
            axs[1].imshow(rotated_slice_map_4, cmap='jet', alpha=0.5)
            axs[1].axis('off')
            axs[1].set_title(str(slice_index_4))

            rotated_original_slice_2 = np.rot90(original_slice_2, k=1)
            rotated_slice_map_2 = np.rot90(slice_map_2, k=1)
            axs[2].imshow(rotated_original_slice_2, cmap='gray', alpha=0.7)
            axs[2].imshow(rotated_slice_map_2, cmap='jet', alpha=0.5)
            axs[2].axis('off')
            axs[2].set_title(str(slice_index_2))

            rotated_original_slice_3 = np.rot90(original_slice_3, k=1)
            rotated_slice_map_3 = np.rot90(slice_map_3, k=3)
            axs[3].imshow(rotated_original_slice_3, cmap='gray', alpha=0.7)
            axs[3].imshow(rotated_slice_map_3, cmap='jet', alpha=0.5)
            axs[3].axis('off')
            axs[3].set_title(str(slice_index_3))

            # 输出文件名：用原始 npy 文件名，避免路径非法字符
            img_base = os.path.splitext(os.path.basename(img_path))[0]
            # save_path = os.path.join(out_dir, f"{img_base}.png")
            save_path = os.path.join(class_out_dir, f"{img_base}.png")
            # plt.show()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)  # 重要：防止内存/句柄累积

        print(f"Saved CAM images to: {out_dir}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
