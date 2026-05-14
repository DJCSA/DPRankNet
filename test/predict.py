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


def get_classes(root: str):
    mri_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    mri_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(mri_class))
    return class_indices


def evaluate(confusion_matrix, pred, gt, n_classes):
    score_all_t = []
    confusion_matrix = torch.from_numpy(confusion_matrix)
    ACC = confusion_matrix.diag().sum() / confusion_matrix.sum()
    Pe = ((confusion_matrix.sum(0) / confusion_matrix.sum()) * (confusion_matrix.sum(1) / confusion_matrix.sum())).sum()
    kappa = 1 - (1 - ACC) / (1 - Pe + 1e-9)

    pred_probs = torch.softmax(pred.cpu(), dim=1).numpy()
    gt_bin = label_binarize(gt.cpu(), classes=[0, 1, 2])
    for i in range(n_classes):
        TP = confusion_matrix[i, i]
        FP = torch.sum(confusion_matrix[:, i]) - TP
        TN = confusion_matrix.diag().sum() - TP
        FN = confusion_matrix.sum() - TP - FP - TN

        SPE = TN / (TN + FP + 1e-10)
        SEN = TP / (TP + FN + 1e-10)
        PRE = TP / (TP + FP + 1e-10)
        F1 = 2 * PRE * SEN / (PRE + SEN)
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-9)
        fpr, tpr, th = roc_curve(gt_bin[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)

        score_all_t.append([SPE, SEN, PRE, F1, MCC, roc_auc])
    metric_mat = np.mean(np.array(score_all_t), axis=0)

    SPE, SEN, PRE, F1, MCC, roc_auc = metric_mat[0], metric_mat[1], metric_mat[2], metric_mat[3], metric_mat[4], \
        metric_mat[5]
    return ACC, SPE, SEN, PRE, F1, MCC, kappa, roc_auc


def evaluate_2(confusion_matrix, TN_ind, pred, gt, n_classes):
    confusion_matrix = torch.from_numpy(confusion_matrix)
    TN = confusion_matrix[TN_ind, TN_ind]
    FN = confusion_matrix[:, TN_ind].sum() - TN
    TP = confusion_matrix.diag().sum() - TN
    FP = confusion_matrix.sum() - TN - FN - TP

    SPE = TN / (TN + FP + 1e-10)
    SEN = TP / (TP + FN + 1e-10)
    PRE = TP / (TP + FP + 1e-10)
    F1 = 2 * PRE * SEN / (PRE + SEN + 1e-10)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-9)

    pred_probs = torch.softmax(pred.cpu(), dim=1).numpy()
    gt_bin = label_binarize(gt.cpu(), classes=[0, 1, 2])
    roc_auc_list = []
    for i in range(n_classes):
        if i == TN_ind: continue
        fpr, tpr, th = roc_curve(gt_bin[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_list.append(roc_auc)

    roc_auc_mean = np.mean(np.array(roc_auc_list), axis=0)
    return SPE, SEN, PRE, F1, MCC, roc_auc_mean


def main():
    classes = get_classes(r"D:\DJC\dataset")
    root_dir = r"D:\DJC\test"  # 根目录
    csv_file = r"D:\DJC\test\hard_pro\results_noisy.csv"  # 统一的结果文件
    test_data_path = r'D:\DJC\Ablation\ADNI_data_notest_age\AIBL.csv'
    batch_size = 12
    num_workers = 0
    cm_meam = np.zeros((3, 3), dtype=np.int32)

    # 获取设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化数据加载器（只需要一次）
    adni_dataloader = Adni_dataloader(test_data_path, batch_size, num_workers, AIBL=True)
    test_dataloader = adni_dataloader.run()

    # 创建CSV并写入表头（如果不存在）
    # if not os.path.exists(csv_file):
    #     with open(csv_file, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         # writer.writerow([
    #         #     'model_path', 'acc', 'SPE', 'SEN', 'PRE', 'F1', 'Kappa',
    #         #     'AUC', 'MCC', 'SPE_2', 'SEN_2', 'PRE_2', 'F1_2', 'AUC_2', 'MCC_2'
    #         # ])
    #         writer.writerow(['model_path', 'Pred_AD', 'Pred_MCI', 'Pred_NC'])

    # 递归搜索所有.pth文件
    pth_files = glob.glob(os.path.join(root_dir, '**', '*.pth'), recursive=True)
    print(f"Found {len(pth_files)} .pth files to process")

    # 遍历每个模型文件
    for pth_path in pth_files:
        try:
            print(f"\nProcessing: {pth_path}")

            # 初始化模型
            model = resnet18().to(device)
            model.load_state_dict(torch.load(pth_path, map_location=device), strict=False)
            model.eval()

            # 重置统计量
            confusion_matrix = np.zeros((3, 3), dtype=np.int32)
            preds_list, gts_list = [], []

            # 推理过程
            with torch.no_grad():
                for images, labels, _ in tqdm(test_dataloader, desc="Testing", file=sys.stdout):
                    images = images.to(device)
                    labels = labels.to(device)

                    pred, _ = model(images)
                    preds_list.append(pred.cpu())
                    gts_list.append(labels.cpu())

                    # 更新混淆矩阵
                    pred_classes = torch.argmax(pred, dim=1)
                    for t, p in zip(labels.cpu().numpy(), pred_classes.cpu().numpy()):
                        confusion_matrix[t, p] += 1

            # 计算指标
            preds_roc = torch.cat(preds_list)
            gts_roc = torch.cat(gts_list)
            # print(confusion_matrix)
            # cm_meam += confusion_matrix
            # print("mean", cm_meam)

            probs = torch.softmax(preds_roc, dim=1)

            # 2. 提取出每个样本【真实标签】对应的预测概率
            total_samples = gts_roc.size(0)
            indices = torch.arange(total_samples)
            # prob_true = probs[indices, gts_roc]

            # 3. 提取出每个样本【其他类别】的最大预测概率
            probs_other = probs.clone()
            probs_other[indices, gts_roc] = -1.0
            # max_other_prob, _ = torch.max(probs_other, dim=1)

            # 4. 根据条件进行全局布尔判断
            # is_easy = prob_true >= 0.7
            # is_hard = (prob_true < 0.7) & (max_other_prob <= 0.6)
            # is_noisy = ~(is_easy | is_hard)

            # 5. 分类别统计数量 (0: AD, 1: MCI, 2: NC)
            # class_names = ['AD', 'MCI', 'NC']
            # stats_dict = {}

            # print("Sample Stats per Class:")
            # for i, c_name in enumerate(class_names):
            #     # 找出真实标签为当前类别的样本掩码
            #     class_mask = (gts_roc == i)

            #     # 结合全局条件和当前类别掩码进行统计 (逻辑与 &)
            #     c_easy = (is_easy & class_mask).sum().item()
            #     c_hard = (is_hard & class_mask).sum().item()
            #     c_noisy = (is_noisy & class_mask).sum().item()

            #     # 存入字典以备写入 CSV
            #     stats_dict[c_name] = {'Easy': c_easy, 'Hard': c_hard, 'Noisy': c_noisy}
            #     print(f"  {c_name} -> Easy: {c_easy}, Hard: {c_hard}, Noisy: {c_noisy}")
            # 逐行写入矩阵数据
            # mci_hard_mask = (gts_roc == 1) & is_hard

            # 2. 从所有的概率张量中提取出这些符合条件的样本概率
            # mci_hard_probs = probs[mci_hard_mask]

            # 3. 将结果写入 CSV
            # with open(csv_file, 'a', newline='') as f:
            #     writer = csv.writer(f)

            #     # 写入当前模型路径作为标识，方便区分不同模型的输出
            #     writer.writerow([f"Model: {os.path.basename(pth_path)}"])

            #     # 如果该模型下有 MCI_hard 样本，则写入具体概率
            #     if mci_hard_probs.size(0) > 0:
            #         writer.writerow(['Sample_Type', 'Prob_AD', 'Prob_MCI', 'Prob_NC'])

            #         # 遍历并写入每一个 MCI hard 样本的三类概率（保留4位小数）
            #         for p in mci_hard_probs:
            #             writer.writerow([
            #                 'MCI_Hard',
            #                 f"{p[0].item():.4f}",  # 预测为 AD 的概率
            #                 f"{p[1].item():.4f}",  # 预测为 MCI 的概率
            #                 f"{p[2].item():.4f}"  # 预测为 NC 的概率
            #             ])
            #     else:
            #         writer.writerow(['No MCI_Hard samples found for this model.'])

            #     # 写入一个空行，作为不同模型之间的视觉分隔
            #     writer.writerow([])
            # with open(csv_file, 'a', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([pth_path])
            #     writer.writerow(['True_AD', confusion_matrix[0, 0], confusion_matrix[0, 1], confusion_matrix[0, 2]])
            #     writer.writerow(['True_MCI', confusion_matrix[1, 0], confusion_matrix[1, 1], confusion_matrix[1, 2]])
            #     writer.writerow(['True_NC', confusion_matrix[2, 0], confusion_matrix[2, 1], confusion_matrix[2, 2]])

                # 3. 写入一个空行，作为不同模型之间的视觉分隔
                # writer.writerow([])
            ACC, SPE, SEN, PRE, F1, MCC, kappa, ROC = evaluate(confusion_matrix, preds_roc, gts_roc, 3)
            SPE_2, SEN_2, PRE_2, F1_2, MCC_2, ROC_2 = evaluate_2(confusion_matrix, 2, preds_roc, gts_roc, 3)

            # 写入结果
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    pth_path,
                    round(ACC.item(), 4),
                    round(SPE.item(), 4),
                    round(SEN.item(), 4),
                    round(PRE.item(), 4),
                    round(F1.item(), 4),
                    round(kappa.item(), 4),
                    round(ROC, 4),
                    round(MCC.item(), 4),
                    round(SPE_2.item(), 4),
                    round(SEN_2.item(), 4),
                    round(PRE_2.item(), 4),
                    round(F1_2.item(), 4),
                    round(ROC_2, 4),
                    round(MCC_2.item(), 4)
                ])

            # for i, key in enumerate(preds_list):
            #     for j, k in enumerate(preds_list[i]):
            #         writer.writerow([
            #             gts_list[i][j].item(),
            #             preds_list[i][j][0].item(),
            #             preds_list[i][j][1].item(),
            #             preds_list[i][j][2].item()
            #         ])

        except Exception as e:
            print(f"Error processing {pth_path}: {str(e)}")
            continue

    print("\nAll models processed. Results saved to:", csv_file)


if __name__ == '__main__':
    main()
