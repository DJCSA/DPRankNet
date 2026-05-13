# coding:utf-8
import matplotlib.pyplot as plt
import torch
# from disentangle_mhsa import resnet18
# from resnet_list import resnet18
from resnet_select import resnet18
# from select_net.resnet_select import resnet18
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import sys
import os
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path


class Get_Dataset(Dataset):

    def __init__(self, data_path, transforms=None, double=None, AIBL=False):
        self.subjects = pd.read_csv(data_path)
        self.index = list(self.subjects.columns.values)
        self.index.pop(0)
        self.transforms = transforms
        self.double = double
        self.AIBL = AIBL

    def __len__(self):
        return len(self.subjects[self.index[0]])

    def __getitem__(self, idx):
        img_path = self.subjects[self.index[0]][idx]
        if self.AIBL is True:
            group = self.subjects[self.index[1]][idx]
            # group = group.long()
            if group == 0:
                img_path = r"D:\DJC\AIBL\AD/" + img_path
            elif group == 1:
                img_path = r"D:\DJC\AIBL\MCI/" + img_path
            else:
                img_path = r"D:\DJC\AIBL\NC/" + img_path
            img = np.load(img_path)
            assert img is not None
            group = self.subjects[self.index[1]][idx]
            img = img.astype(np.float32)
            tensor_data = torch.from_numpy(img)
            tensor_data = tensor_data.unsqueeze(0)
            label = group
        else:
            img_path1 = img_path[23:-18]
            img_path2 = img_path[-11:]
            img_path = r"D:\DJC/" + img_path1 + img_path2
            # print(img_path)
            # print(idx)
            img = np.load(img_path)
            assert img is not None
            group = self.subjects[self.index[1]][idx]
            img = img.astype(np.float32)
            tensor_data = torch.from_numpy(img)
            label = group

        return tensor_data, label, idx


def get_class_name(root: str):
    mri_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    mri_class.sort()
    class_indices = [v for v, k in enumerate(mri_class)]
    class_indices_dict = dict((v, k) for v, k in enumerate(mri_class))
    return class_indices, class_indices_dict


features = []


def hook_fn(module, input, output):
    features.append(output.clone().detach())


def mean_all(features):
    for step, i in enumerate(features):  # element--->b,c,h,w
        temp = torch.mean(i.view(i.size(0), i.size(1), -1), dim=2)
        if step == 0:
            result = temp.detach()
        else:
            result = torch.cat((result, temp), dim=0)
    return result.cpu().numpy()


def process_fold(fold_path, device):
    """处理单个fold目录下的所有权重"""
    fold_path = Path(fold_path)
    weight_files = sorted(fold_path.glob("*.pth"))

    for weight_file in weight_files:
        global features
        features = []

        # 创建模型
        model = resnet18().to(device)
        hook = model.maxpool.register_forward_hook(hook_fn)

        # 加载权重
        print(f"\nLoading weights: {weight_file.name}")
        model.load_state_dict(torch.load(weight_file, map_location=device))

        # 数据加载
        test_dataset = Get_Dataset(r"D:\DJC\Ablation\ADNI_data_notest_age\AIBL.csv", AIBL=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=12,
            num_workers=0,
            shuffle=False
        )

        # 特征提取
        model.eval()
        avg_features = []
        all_probs_true = []
        all_probs_other_max = []
        all_labels_list = []  # 【新增】用于存储所有样本的真实标签，以便后续按类别统计

        with torch.no_grad():
            for images, labels, _ in tqdm(test_loader):
                images = images.to(device)

                logits, outputs = model(images)
                avg_features.append(outputs.cpu().numpy())

                probs = torch.softmax(logits, dim=1)
                labels_idx = labels.to(device).long()

                # 【新增】记录真实标签
                all_labels_list.extend(labels_idx.cpu().numpy())

                batch_size = images.size(0)
                batch_indices = torch.arange(batch_size)

                try:
                    # 1. 提取真实标签对应的概率
                    prob_true = probs[batch_indices, labels_idx]
                    all_probs_true.extend(prob_true.cpu().numpy())

                    # 2. 提取除真实标签外，其余标签中的最大概率
                    probs_other = probs.clone()
                    probs_other[batch_indices, labels_idx] = -1.0  # 将真实标签概率置为-1，排除干扰
                    prob_other_max, _ = probs_other.max(dim=1)
                    all_probs_other_max.extend(prob_other_max.cpu().numpy())
                except Exception as e:
                    raise ValueError("请确保 DataLoader 返回的 labels 是能够对应模型输出的整数类别索引。") from e

        # 处理特征
        avg_features = np.concatenate(avg_features, axis=0)
        all_probs_true = np.array(all_probs_true)
        all_probs_other_max = np.array(all_probs_other_max)
        all_labels_array = np.array(all_labels_list)  # 【新增】转换为numpy数组

        # 构建各个类别的布尔掩码 (Mask)
        # Easy: 真实标签预测概率 >= 0.7
        mask_easy = all_probs_true >= 0.7
        # Hard: 真实标签预测概率 < 0.7 且 其他标签中有概率 <= 0.6 (根据你最新代码保留)
        mask_hard = (all_probs_true < 0.7) & (all_probs_other_max <= 0.6)
        # Noisy: 既不是 Easy 也不是 Hard
        mask_noisy = ~(mask_easy | mask_hard)

        print("\n================ 特征分类统计 ================")
        print(f"【总体统计】 (总样本数: {len(avg_features)})")
        print(f"  - Easy  总数: {mask_easy.sum():>4} ({mask_easy.sum() / len(avg_features) * 100:>5.2f}%)")
        print(f"  - Hard  总数: {mask_hard.sum():>4} ({mask_hard.sum() / len(avg_features) * 100:>5.2f}%)")
        print(f"  - Noisy 总数: {mask_noisy.sum():>4} ({mask_noisy.sum() / len(avg_features) * 100:>5.2f}%)")

        # 【新增】按类别统计各个集合的数量
        print("\n【按类别详细统计】")
        # 获取类别字典，将数字索引映射为类别名称 (AD, MCI, NC等)
        _, class_dict = get_class_name(r"D:\DJC\dataset")

        # 遍历数据集中实际存在的所有类别标签
        for cls_idx in np.unique(all_labels_array):
            cls_name = class_dict.get(cls_idx, f"Class_{cls_idx}")

            # 当前类别的掩码 (例如：所有标签为 AD 的样本)
            cls_mask = (all_labels_array == cls_idx)
            total_c = cls_mask.sum()

            # 计算当前类别下，分别属于 Easy, Hard, Noisy 的数量
            easy_c = (mask_easy & cls_mask).sum()
            hard_c = (mask_hard & cls_mask).sum()
            noisy_c = (mask_noisy & cls_mask).sum()

            print(f"▷ 类别: {cls_name} (总数: {total_c})")
            print(f"    ├─ Easy  : {easy_c:>4} ({easy_c / total_c * 100:>5.2f}%)")
            print(f"    ├─ Hard  : {hard_c:>4} ({hard_c / total_c * 100:>5.2f}%)")
            print(f"    └─ Noisy : {noisy_c:>4} ({noisy_c / total_c * 100:>5.2f}%)")
        print("==============================================\n")

        # 定义要绘制的图及其对应的掩码
        plot_configs = [
            ("hard_only", mask_hard),
            ("noisy_and_easy", mask_noisy | mask_easy),
            ("noisy_only", mask_noisy),
            ("hard_and_easy", mask_hard | mask_easy),
            ("easy_only", mask_easy),
        ]

        # 分别绘制图
        for plot_name, mask in plot_configs:
            filtered_features = avg_features[mask]

            if len(filtered_features) < 2:
                print(f"⚠️ 警告: [{plot_name}] 的样本数量少于2个，无法进行t-SNE绘制，跳过！")
                continue

            print(f"正在生成图像: {plot_name} (包含 {len(filtered_features)} 个样本)...")
            generate_tsne(
                features=filtered_features,
                meta_path=r"D:\DJC\Ablation\ADNI_data_notest_age\AIBL.csv",
                save_dir=fold_path.parent / "tsne_results2" / fold_path.name,
                weight_name=f"{weight_file.stem}_{plot_name}",
                valid_mask=mask
            )

        # 清理资源
        hook.remove()
        del model
        torch.cuda.empty_cache()


def generate_tsne(features, meta_path, save_dir, weight_name, valid_mask=None):
    """原始可视化逻辑封装"""
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载元数据
    df = pd.read_csv(meta_path)

    # 同步过滤 DataFrame，保持与 features 的长度一致
    if valid_mask is not None:
        df = df.iloc[valid_mask].reset_index(drop=True)

    classes, class_dict = get_class_name(r"D:\DJC\dataset")

    # 动态调整 perplexity，防止样本过少时 t-SNE 报错
    n_samples = features.shape[0]
    perplexity_value = min(30, max(1, n_samples - 1))

    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity_value)
    tsne_results = tsne.fit_transform(features)

    # 保持原始绘图参数
    plt.figure(figsize=(13, 13))
    mark_list = ['.', '.', '.']
    palette = sns.hls_palette(len(classes))
    edge = np.array([[-20, -20], [20, -20], [20, 20], [-20, 20]])

    # 绘制每个类别
    for idx, value in enumerate(classes):
        color = palette[idx]
        mark = mark_list[idx]
        indices = np.where(df['class'] == value)[0]

        # 防止某一类全部被过滤掉导致空索引报错
        if len(indices) > 0:
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=color, marker=mark,
                        label=class_dict[value], s=250)

    plt.scatter(edge[:, 0], edge[:, 1], color='w')
    plt.legend(fontsize=16, markerscale=3, bbox_to_anchor=(1, 1))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # 保存结果
    plt.savefig(save_dir / f"tsne_{weight_name}.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_dir = Path(r"D:\DJC\Ablation\t_sne\we")

    # 遍历所有fold目录
    for fold_dir in base_dir.glob("*"):
        if fold_dir.is_dir():
            print(f"\nProcessing {fold_dir.name}...")
            process_fold(fold_dir, device)


if __name__ == "__main__":
    main()