import csv
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import sys
from dataloder_predict import *
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from resnet_select import resnet18
import faiss
import torch.nn.functional as F


def get_classes(root: str):
    mri_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    mri_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(mri_class))
    return class_indices


def compute_euclidean_distances(features1, features2):
    """
    计算两个特征集之间的欧式距离

    Args:
        features1: 第一个特征集, shape [n1, feature_dim]
        features2: 第二个特征集, shape [n2, feature_dim]

    Returns:
        distances: 距离矩阵, shape [n1, n2]
        mean_distance: 平均距离
        std_distance: 距离标准差
    """
    # 确保输入是numpy数组
    if isinstance(features1, torch.Tensor):
        features1 = features1.cpu().numpy()
    if isinstance(features2, torch.Tensor):
        features2 = features2.cpu().numpy()

    # 计算欧式距离矩阵
    distances = np.sqrt(np.sum((features1[:, np.newaxis, :] - features2[np.newaxis, :, :]) ** 2, axis=2))

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    return distances, mean_distance, std_distance


def extract_features(model, dataloader, device):
    """
    从数据加载器中提取所有特征

    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备

    Returns:
        all_features: 所有特征, shape [n_samples, feature_dim]
        all_labels: 所有标签
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            pred, features = model(images)

            all_features.append(features.cpu())
            all_labels.append(labels)

    # 合并所有批次的特征
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_features, all_labels


def main():
    classes = get_classes(r"D:\DJC\dataset")
    root_dir = r"D:\DJC\weights_mri\weights_mri\baseline\baseline_mri"  # 根目录
    csv_file = r"D:\DJC\Ablation\ADNI_data_notest_age\results_distance_ba.csv"  # 统一的结果文件
    AD_data_path = r'D:\DJC\Ablation\ADNI_data_notest_age\ADNI_AD.csv'
    MCI_data_path = r'D:\DJC\Ablation\ADNI_data_notest_age\ADNI_MCI.csv'
    NC_data_path = r'D:\DJC\Ablation\ADNI_data_notest_age\ADNI_NC.csv'
    batch_size = 10
    num_workers = 0

    # 获取设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建结果文件（如果不存在）
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'pth', 'AD_AD_mean', 'AD_AD_std', 'AD_MCI_mean', 'AD_MCI_std',
                'AD_NC_mean', 'AD_NC_std', 'MCI_MCI_mean', 'MCI_MCI_std',
                'MCI_NC_mean', 'MCI_NC_std', 'NC_NC_mean', 'NC_NC_std'
            ])

    # 创建数据加载器
    ad_dataloader = Adni_dataloader(AD_data_path, batch_size, num_workers, AIBL=False)
    AD_dataloader = ad_dataloader.run()

    mci_dataloader = Adni_dataloader(MCI_data_path, batch_size, num_workers, AIBL=False)
    MCI_dataloader = mci_dataloader.run()

    nc_dataloader = Adni_dataloader(NC_data_path, batch_size, num_workers, AIBL=False)
    NC_dataloader = nc_dataloader.run()

    # 递归搜索所有.pth文件
    pth_files = glob.glob(os.path.join(root_dir, '**', '*.pth'), recursive=True)
    print(f"Found {len(pth_files)} .pth files to process")

    # 遍历每个模型文件
    for pth_path in pth_files:
        print(f"\nProcessing: {pth_path}")

        # 初始化模型
        model = resnet18().to(device)
        model.load_state_dict(torch.load(pth_path, map_location=device), strict=False)

        # 提取所有特征
        print("Extracting AD features...")
        AD_features, AD_labels = extract_features(model, AD_dataloader, device)

        print("Extracting MCI features...")
        MCI_features, MCI_labels = extract_features(model, MCI_dataloader, device)

        print("Extracting NC features...")
        NC_features, NC_labels = extract_features(model, NC_dataloader, device)

        print(f"Feature shapes - AD: {AD_features.shape}, MCI: {MCI_features.shape}, NC: {NC_features.shape}")

        # 计算各类别之间的欧式距离
        print("Computing Euclidean distances...")

        # AD与AD之间的距离（类内距离）
        AD_AD_dist, AD_AD_mean, AD_AD_std = compute_euclidean_distances(AD_features, AD_features)

        # AD与MCI之间的距离
        AD_MCI_dist, AD_MCI_mean, AD_MCI_std = compute_euclidean_distances(AD_features, MCI_features)

        # AD与NC之间的距离
        AD_NC_dist, AD_NC_mean, AD_NC_std = compute_euclidean_distances(AD_features, NC_features)

        # MCI与MCI之间的距离（类内距离）
        MCI_MCI_dist, MCI_MCI_mean, MCI_MCI_std = compute_euclidean_distances(MCI_features, MCI_features)

        # MCI与NC之间的距离
        MCI_NC_dist, MCI_NC_mean, MCI_NC_std = compute_euclidean_distances(MCI_features, NC_features)

        # NC与NC之间的距离（类内距离）
        NC_NC_dist, NC_NC_mean, NC_NC_std = compute_euclidean_distances(NC_features, NC_features)

        # 打印结果
        print(f"AD-AD: mean={AD_AD_mean:.4f}, std={AD_AD_std:.4f}")
        print(f"AD-MCI: mean={AD_MCI_mean:.4f}, std={AD_MCI_std:.4f}")
        print(f"AD-NC: mean={AD_NC_mean:.4f}, std={AD_NC_std:.4f}")
        print(f"MCI-MCI: mean={MCI_MCI_mean:.4f}, std={MCI_MCI_std:.4f}")
        print(f"MCI-NC: mean={MCI_NC_mean:.4f}, std={MCI_NC_std:.4f}")
        print(f"NC-NC: mean={NC_NC_mean:.4f}, std={NC_NC_std:.4f}")

        # 保存结果到CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                pth_path,
                AD_AD_mean, AD_AD_std,
                AD_MCI_mean, AD_MCI_std,
                AD_NC_mean, AD_NC_std,
                MCI_MCI_mean, MCI_MCI_std,
                MCI_NC_mean, MCI_NC_std,
                NC_NC_mean, NC_NC_std
            ])


if __name__ == '__main__':
    main()