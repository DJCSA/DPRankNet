import argparse
import csv
import math
import os
import sys
from datetime import datetime

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from dataloder_adni import Adni_dataloader
from resnet_select import resnet18

start = datetime.now()


def train_one_epoch_warmup(model, optimizer, data_loader, device, epoch, classes):
    model.train()
    CE_loss = torch.zeros(1).to(device)  # 累计损失
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    class_key = dict((v, k) for v, k in enumerate(classes))
    confusion_matrix = np.zeros(shape=(3, 3), dtype=np.int16)

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    preds_roc, gts_roc = None, None
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, images_aug, labels, _ = data
        labels = labels.long()
        labels_aug = labels
        images = torch.cat([images, images_aug], dim=0).to(device)
        labels_all = torch.cat([labels, labels_aug], dim=0).to(device)
        sample_num += images.shape[0]
        pred, fx = model(images)
        pred_ = pred
        if preds_roc == None:
            preds_roc = pred_
            gts_roc = labels
        else:
            preds_roc = torch.cat([preds_roc, pred_], 0)
            gts_roc = torch.cat([gts_roc, labels], 0)
        pred_classes = torch.max(pred, dim=1)[1]
        for label, prediction in zip(labels_all, pred_classes):
            label = label.to(device)
            confusion_matrix[prediction.item()][label.item()] += 1
            if label == prediction:
                correct_pred[class_key[label.item()]] += 1
            total_pred[class_key[label.item()]] += 1

        accu_num += torch.eq(pred_classes, labels_all.to(device)).sum()
        loss1 = F.cross_entropy(pred, labels_all)
        loss = loss1
        # print(type(loss), loss)
        loss.requires_grad_(True)
        loss.backward()
        accu_loss += loss.detach()
        CE_loss += loss1.detach()

        data_loader.desc = "epoch {}/200 [train] lossCE: {:.3f}, acc: {:.3f}". \
            format(epoch, float(CE_loss / (step + 1)),
                   accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
    print('confusion_matrix: ')
    print(confusion_matrix)

    accuracy_dict = {}
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        accuracy_dict[classname] = accuracy

    return accu_loss.item() / (
            step + 1), accu_num.item() / sample_num, accuracy_dict, confusion_matrix, preds_roc, gts_roc


@torch.no_grad()
def evaluate_warmup(model, data_loader, device, epoch, classes):
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    CE_loss = torch.zeros(1).to(device)  # 累计损失
    confusion_matrix = np.zeros(shape=(3, 3), dtype=np.int16)
    # confusion_matrix_2 = np.zeros(shape=(2, 2), dtype=np.int16)
    if classes != None:
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        class_key = dict((v, k) for v, k in enumerate(classes))
    preds_roc, gts_roc = None, None
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.long()
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred, fx = model(images)
        pred_ = pred
        if preds_roc == None:
            preds_roc = pred_
            gts_roc = labels
        else:
            preds_roc = torch.cat([preds_roc, pred_], 0)
            gts_roc = torch.cat([gts_roc, labels], 0)

        pred_num = torch.max(F.softmax(pred, dim=1), dim=1)[0].detach()
        pred_classes = torch.max(pred, dim=1)[1]

        if classes is not None:
            for label, prediction in zip(labels, pred_classes):
                # print('label:',label.item())
                # print('prediction',prediction.item())
                label = label.to(device)
                confusion_matrix[prediction.item()][label.item()] += 1

                if label == prediction:
                    correct_pred[class_key[label.item()]] += 1
                total_pred[class_key[label.item()]] += 1

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss1 = F.cross_entropy(pred, labels)
        loss = loss1
        accu_loss += loss
        CE_loss += loss1

        data_loader.desc = "[validation] lossCE: {:.3f}, acc: {:.3f}". \
            format(float(CE_loss / (step + 1)),
                   accu_num.item() / sample_num)
    if classes != None:

        accuracy_dict = {}
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            # print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
            accuracy_dict[classname] = accuracy
        return accu_loss.item() / (
                step + 1), accu_num.item() / sample_num, accuracy_dict, confusion_matrix, preds_roc, gts_roc
    # 有两种可能，1.我的测试数据没有随机打乱，划分有问题 2.这个显示的代码设置有问题
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num  # ,accuracy_dict


@torch.no_grad()
def evaluate_listwise(model, data_loader, device, epoch, classes):
    # loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    CE_loss = torch.zeros(1).to(device)  # 累计损失
    # triplet_loss = torch.zeros(1).to(device)  # 累计损失
    list_wise_loss_all = torch.zeros(1).to(device)  # 累计损失
    confusion_matrix = np.zeros(shape=(3, 3), dtype=np.int16)

    preds_list = []
    gts_list = []
    if classes != None:
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        class_key = dict((v, k) for v, k in enumerate(classes))

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.long()
        # print(labels)
        sample_num += images.shape[0]
        images = images.to(device)
        labels = labels.to(device)

        pred, fx = model(images)

        preds_list.append(pred.cpu())
        gts_list.append(labels.cpu())

        pred_num = torch.max(F.softmax(pred, dim=1), dim=1)[0].detach()
        pred_classes = torch.max(pred, dim=1)[1]

        if classes != None:
            for label, prediction in zip(labels, pred_classes):
                # print('label:',label.item())
                # print('prediction',prediction.item())
                label = label.to(device)
                confusion_matrix[prediction.item()][label.item()] += 1
                if label == prediction:
                    correct_pred[class_key[label.item()]] += 1
                total_pred[class_key[label.item()]] += 1

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss1 = F.cross_entropy(pred, labels)
        # list_wise_loss = get_list_wise_loss(F.softmax(pred, dim=1), labels)

        # loss_l = get_new_listwise_loss(fx, labels, device)
        loss = loss1
        # print(type(loss), loss)
        accu_loss += loss
        CE_loss += loss1
        # triplet_loss += 0.001 * loss_triplet
        # list_wise_loss_all += 0.01 * loss_l

        data_loader.desc = "[validation] loss: {:.3f} , acc: {:.3f}". \
            format(float(CE_loss / (step + 1)),
                   accu_num.item() / sample_num)
    if classes != None:
        # print('total_pred_end', total_pred)
        # print('correct_pred_end', correct_pred)

        accuracy_dict = {}
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            # print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
            accuracy_dict[classname] = accuracy
        return accu_loss.item() / (
                step + 1), accu_num.item() / sample_num, accuracy_dict, confusion_matrix, preds_list, gts_list
    # 有两种可能，1.我的测试数据没有随机打乱，划分有问题 2.这个显示的代码设置有问题
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num  # ,accuracy_dict


def get_ssl(feat, feat_aug, batch_size, temperature):
    sim_clean = torch.mm(feat, feat.t())
    mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
    sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

    sim_aug = torch.mm(feat, feat_aug.t())
    sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)

    logits_pos = torch.bmm(feat.view(batch_size, 1, -1), feat_aug.view(batch_size, -1, 1)).squeeze(-1)
    logits_neg = torch.cat([sim_clean, sim_aug], dim=1)

    logits = torch.cat([logits_pos, logits_neg], dim=1)
    instance_labels = torch.zeros(batch_size).long().cuda()

    loss_instance = F.cross_entropy(logits / temperature, instance_labels)

    return loss_instance


def get_scl_2_(feat, feat_aug, labels, temperature, device):
    feat_ad = feat[labels == 0]  # c1
    feat_mci = feat[labels == 1]  # c2
    feat_nc = feat[labels == 2]  # c3
    feat_ad_aug = feat_aug[labels == 0]  # c1
    feat_mci_aug = feat_aug[labels == 1]  # c2
    feat_nc_aug = feat_aug[labels == 2]  # c3

    sim_aug_ad = torch.mm(feat_ad, torch.cat((feat_mci, feat_mci_aug, feat_nc, feat_nc_aug), dim=0).t())  # c1,c2233
    sim_aug_mci = torch.mm(feat_mci, torch.cat((feat_ad, feat_ad_aug, feat_nc, feat_nc_aug), dim=0).t())  # c2,c1133
    sim_aug_nc = torch.mm(feat_nc, torch.cat((feat_ad, feat_ad_aug, feat_mci, feat_mci_aug), dim=0).t())  # c3,c11c22

    loss = torch.tensor([0.]).to(device)  # .cuda()
    if feat_ad.size(0) > 0:
        logits_pos_ad = torch.bmm(feat_ad.view(feat_ad.size(0), 1, -1),
                                  feat_ad_aug.view(feat_ad.size(0), -1, 1)).squeeze(-1)  # c1
        logits_ad = torch.cat([logits_pos_ad, sim_aug_ad], dim=1)
        instance_labels_ad = torch.zeros(feat_ad.size(0)).long().to(device)  # .cuda()
        loss_ad = F.cross_entropy(logits_ad / temperature, instance_labels_ad)
        loss += loss_ad

    if feat_mci.size(0) > 0:
        logits_pos_mci = torch.bmm(feat_mci.view(feat_mci.size(0), 1, -1),
                                   feat_mci_aug.view(feat_mci.size(0), -1, 1)).squeeze(-1)  # c2
        logits_mci = torch.cat([logits_pos_mci, sim_aug_mci], dim=1)
        instance_labels_mci = torch.zeros(feat_mci.size(0)).long().to(device)  # .cuda()
        loss_mci = F.cross_entropy(logits_mci / temperature, instance_labels_mci)
        loss += loss_mci

    if feat_nc.size(0) > 0:
        logits_pos_nc = torch.bmm(feat_nc.view(feat_nc.size(0), 1, -1),
                                  feat_nc_aug.view(feat_nc.size(0), -1, 1)).squeeze(-1)  # c3
        logits_nc = torch.cat([logits_pos_nc, sim_aug_nc], dim=1)
        instance_labels_nc = torch.zeros(feat_nc.size(0)).long().to(device)  # .cuda()
        loss_nc = F.cross_entropy(logits_nc / temperature, instance_labels_nc)
        loss += loss_nc

    del feat_ad, feat_ad_aug, feat_mci, feat_mci_aug, feat_nc, feat_nc_aug
    del sim_aug_ad, sim_aug_mci, sim_aug_nc

    ssl_loss = loss / 3.

    return ssl_loss


def get_pl_loss(fx, labels, device):
    distance = torch.zeros(size=(fx.shape[0], fx.shape[0])).to(device)
    for r in range(fx.shape[0]):
        for c in range(fx.shape[0]):
            distance[r][c] = torch.sum((fx[r] - fx[c]) ** 2)

    classes = []
    for c in range(3):
        classes.append(torch.argwhere(labels[:int(fx.shape[0] / 2)] == c))
    if len(classes) != 3:
        return torch.tensor(1e-9)

    avgs = []
    half = int(fx.shape[0] / 2)
    for i in range(classes[0].shape[0]):
        for j in range(classes[1].shape[0]):
            for k in range(classes[2].shape[0]):
                ad, mci, nc = classes[0][i], classes[1][j], classes[2][k]
                ad_aug, mci_aug, nc_aug = ad + half, mci + half, nc + half
                avg1 = max(distance[ad, ad_aug], distance[mci, mci_aug], distance[nc, nc_aug])
                avg2 = min(distance[ad, mci], distance[ad, mci_aug], distance[ad_aug, mci], distance[ad_aug, mci_aug],
                           distance[nc, mci], distance[nc, mci_aug], distance[nc_aug, mci], distance[
                               nc_aug, mci_aug])
                avg3 = max(distance[ad, mci], distance[ad, mci_aug], distance[ad_aug, mci], distance[ad_aug, mci_aug],
                           distance[nc, mci], distance[nc, mci_aug], distance[nc_aug, mci], distance[
                               nc_aug, mci_aug])
                avg4 = min(distance[ad, nc], distance[ad, nc_aug], distance[ad_aug, nc], distance[ad_aug, nc_aug])
                avgs.append([avg1, avg2, avg3, avg4])
                # print(avg1,avg2,avg3,avg4)
    loss = torch.zeros(1).to(device)
    pl_loss = torch.zeros(1).to(device)
    for i in range(len(avgs)):
        loss -= torch.log(
            (avgs[i][3] / (avgs[i][0] + avgs[i][1] + avgs[i][2] + avgs[i][3]))
            * (avgs[i][2] / (avgs[i][0] + avgs[i][1] + avgs[i][2]))
            * (avgs[i][1] / (avgs[i][0] + avgs[i][1])))
        # print(float(loss))
    if len(avgs) > 0:
        pl_loss = (loss + 1e-9) / float(len(avgs))

    return pl_loss


def get_pl_loss_a(fx, labels, device):
    distance = torch.zeros(size=(fx.shape[0], fx.shape[0])).to(device)
    for r in range(fx.shape[0]):
        for c in range(fx.shape[0]):
            distance[r][c] = torch.sum((fx[r] - fx[c]) ** 2)

    classes = []
    for c in range(3):
        classes.append(torch.argwhere(labels[:int(fx.shape[0] / 2)] == c))
    if len(classes) != 3:
        return torch.tensor(1e-9)
    avgs = []
    half = int(fx.shape[0] / 2)
    for i in range(classes[0].shape[0]):
        for j in range(classes[1].shape[0]):
            for k in range(classes[2].shape[0]):
                ad, mci, nc = classes[0][i], classes[1][j], classes[2][k]
                ad_aug, mci_aug, nc_aug = ad + half, mci + half, nc + half
                avg1 = (distance[ad, ad_aug] + distance[mci, mci_aug] + distance[nc, nc_aug]) / 3
                avg2 = (distance[ad, mci] + distance[ad, mci_aug] + distance[ad_aug, mci] + distance[ad_aug, mci_aug] +
                        distance[nc, mci] + distance[nc, mci_aug] + distance[nc_aug, mci] + distance[
                            nc_aug, mci_aug]) / 8
                avg3 = (distance[ad, nc] + distance[ad, nc_aug] + distance[ad_aug, nc] + distance[ad_aug, nc_aug]) / 4
                avgs.append([avg1, avg2, avg3])
    loss = torch.zeros(1).to(device)
    for i in range(len(avgs)):
        loss -= torch.log(
            (avgs[i][2] / (avgs[i][0] + avgs[i][1] + avgs[i][2])) * (avgs[i][1] / (avgs[i][0] + avgs[i][1])))
    return loss


def get_triplet_loss(fx, labels, clean_idx, other_idx, device, alpha=0.5):
    # 提取clean和other样本索引
    clean_samples = torch.where(clean_idx)[0]  # Tensor形式
    other_samples = torch.where(other_idx)[0]

    # 合并所有有效样本
    valid_samples = torch.cat([clean_samples, other_samples])
    fx_valid = fx[valid_samples]
    labels_valid = labels[valid_samples]

    # 按类别统计数量（保持原始逻辑）
    num_classes = 3
    num_con = torch.zeros(num_classes, dtype=torch.long, device=device)
    for c in range(num_classes):
        num_con[c] = (labels_valid[:len(clean_samples)] == c).sum()

    num_other = torch.zeros(num_classes, dtype=torch.long, device=device)
    for c in range(num_classes):
        num_other[c] = (labels_valid[len(clean_samples):] == c).sum()

    # 初始化损失张量
    loss = torch.tensor(0.0, device=fx.device, requires_grad=True)

    # 类别0处理（锚点来自clean的0类，正样本来自other的0类，负样本来自clean的1类）
    start = 0
    end = num_con[0]
    for i in range(start, end):  # clean中的类别0锚点
        for j in range(len(clean_samples), len(clean_samples) + num_other[0]):  # other中的类别0正样本
            for u in range(num_con[0], num_con[0] + num_con[1]):  # clean中的类别1负样本
                d_positive = torch.norm(fx_valid[i] - fx_valid[j], p=2)
                d_negative = torch.norm(fx_valid[i] - fx_valid[u], p=2)
                current_loss = torch.relu(d_positive - d_negative + alpha)
                loss = loss + current_loss
                # loss += torch.relu(d_positive - d_negative + alpha)

    # 类别1处理（锚点来自clean的1类，正样本来自other的1类，负样本来自clean的0或2类）
    start = num_con[0]
    end = num_con[0] + num_con[1]
    for i in range(start, end):  # clean中的类别1锚点
        for j in range(len(clean_samples) + num_other[0],
                       len(clean_samples) + num_other[0] + num_other[1]):  # other中的类别1正样本
            # 负样本1：clean中的类别0
            for u in range(0, num_con[0]):
                d_positive = torch.norm(fx_valid[i] - fx_valid[j], p=2)
                d_negative = torch.norm(fx_valid[i] - fx_valid[u], p=2)
                # loss += torch.relu(d_positive - d_negative + alpha)
                current_loss = torch.relu(d_positive - d_negative + alpha)
                loss = loss + current_loss

            # 负样本2：clean中的类别2
            for u in range(num_con[0] + num_con[1], len(clean_samples)):
                d_positive = torch.norm(fx_valid[i] - fx_valid[j], p=2)
                d_negative = torch.norm(fx_valid[i] - fx_valid[u], p=2)
                # loss += torch.relu(d_positive - d_negative + alpha)
                current_loss = torch.relu(d_positive - d_negative + alpha)
                loss = loss + current_loss

    # 类别2处理（锚点来自clean的2类，正样本来自other的2类，负样本来自clean的1类）
    start = num_con[0] + num_con[1]
    end = len(clean_samples)
    for i in range(start, end):  # clean中的类别2锚点
        for j in range(len(clean_samples) + num_other[0] + num_other[1], len(valid_samples)):  # other中的类别2正样本
            for u in range(num_con[0], num_con[0] + num_con[1]):  # clean中的类别1负样本
                d_positive = torch.norm(fx_valid[i] - fx_valid[j], p=2)
                d_negative = torch.norm(fx_valid[i] - fx_valid[u], p=2)
                # loss += torch.relu(d_positive - d_negative + alpha)
                current_loss = torch.relu(d_positive - d_negative + alpha)
                loss = loss + current_loss

    return loss


def unimodel_loss(label, p, batchsize, device):
    loss = torch.zeros(1).to(device)
    for n in range(batchsize):
        for j in range(2):
            p_1 = p[n, j]
            p_2 = p[n, j + 1]
            loss = loss + max(0, -(p_1 - p_2) * torch.sign(j - label[n]))

    if batchsize > 0:
        loss = (loss / batchsize)

    return loss


def train_one_epoch_select(model, optimizer, data_loader, device, epoch, classes, clean_idx, hard_idx, noisy_idx,
                           temperature, w_pl_a, w_pl_m, w_uni, w_tri):
    model.train()

    CE_loss = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    pl_a_all = torch.zeros(1).to(device)
    pl_m_all = torch.zeros(1).to(device)
    triplet_loss = torch.zeros(1).to(device)
    # scl_loss = torch.zeros(1).to(device)
    uni_loss = torch.zeros(1).to(device)
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    class_key = dict((v, k) for v, k in enumerate(classes))
    confusion_matrix = np.zeros(shape=(3, 3), dtype=np.int16)

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    preds_list = []
    gts_list = []
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, images_aug, labels, batch_idx = data
        labels = labels.long()
        labels_aug = labels
        images_all = torch.cat([images, images_aug], dim=0).to(device)
        labels_all = torch.cat([labels, labels_aug]).to(device)

        idx_clean = torch.cat([clean_idx[batch_idx], clean_idx[batch_idx]])
        idx_hard = torch.cat([hard_idx[batch_idx], hard_idx[batch_idx]])
        # idx_noisy = torch.cat([noisy_idx[batch_idx], noisy_idx[batch_idx]])
        combined_mask = idx_clean | idx_hard
        num_samples = combined_mask.sum().item()

        sample_num += (images.shape[0] * 2)
        pred, fx, fx_ssl = model(images_all, True)
        preds_list.append(pred.cpu())
        gts_list.append(labels.cpu())

        pred_classes = torch.max(pred, dim=1)[1]

        for label, prediction in zip(labels_all, pred_classes):
            label = label.to(device)
            confusion_matrix[prediction.item()][label.item()] += 1

            if label == prediction:
                correct_pred[class_key[label.item()]] += 1
            total_pred[class_key[label.item()]] += 1

        accu_num += torch.eq(pred_classes, labels_all.to(device)).sum()

        if (idx_clean | idx_hard == False).all():
            loss_ce = torch.zeros(1).to(device)
        else:
            loss_ce = F.cross_entropy(pred[idx_clean | idx_hard], labels_all[idx_clean | idx_hard])
        # loss_ce = F.cross_entropy(pred, labels_all)
        unimodelloss = unimodel_loss(labels_all[idx_clean | idx_hard], pred[idx_clean | idx_hard], num_samples, device)
        # print(num_samples)
        # print(idx_clean)
        # print(idx_hard)
        loss_pl_m = get_pl_loss(fx[idx_clean], labels_all[idx_clean], device)
        loss_pl_a = get_pl_loss_a(fx[idx_clean], labels_all[idx_clean], device)
        # loss_triplet = get_triplet_loss(fx, labels_all, idx_clean, idx_hard, alpha=0.2, device=device)
        del images_all, labels_all, pred, pred_classes, fx
        # torch.cuda.empty_cache()
        # loss_scl = get_scl_2_(fx_ssl[:len(batch_idx)][clean_idx[batch_idx] | hard_idx[batch_idx]],
        #                       fx_ssl[len(batch_idx):][clean_idx[batch_idx] | hard_idx[batch_idx]],
        #                       labels[clean_idx[batch_idx] | hard_idx[batch_idx]], temperature, device)
        del idx_clean, idx_hard, fx_ssl

        # loss = loss_ce + w_tri * loss_triplet
        loss = loss_ce + w_pl_m * loss_pl_m + w_pl_a * loss_pl_a + w_uni * unimodelloss
        # loss = loss_ce + w_pl_m * loss_pl_m + w_pl_a * loss_pl_a + w_tri * loss_triplet + w_scl * loss_scl
        loss.requires_grad_(True)
        loss.backward()
        accu_loss += loss.detach()
        CE_loss += loss_ce.detach()
        pl_a_all += w_pl_a * loss_pl_a.detach()
        pl_m_all += w_pl_m * loss_pl_m.detach()
        # triplet_loss += w_tri * loss_triplet.detach()
        # scl_loss += w_scl * loss_scl.detach()
        uni_loss += w_uni * unimodelloss.detach()
        data_loader.desc = "epoch {} [train] CE: {:.3f} acc: {:.3f}". \
            format(epoch, float(CE_loss / (step + 1)),
                   accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        del loss_ce, loss, loss_pl_a, loss_pl_m, unimodelloss
        torch.cuda.empty_cache()
    accuracy_dict = {}
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        # print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
        accuracy_dict[classname] = accuracy

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, \
        float(CE_loss / (step + 1)), accuracy_dict, float(pl_a_all / (step + 1)), float(pl_m_all / (step + 1)), float(uni_loss / (step + 1)), confusion_matrix, preds_list, gts_list


def compute_features(dataloader, model, N, device):
    print('Compute features')
    model.eval()
    batch_size = dataloader.batch_size
    for i, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = batch[0].to(device)
            output, feat = model(inputs)
            feat = feat.data.cpu().numpy()
            prob = F.softmax(output, dim=1)  # (batch,classes)
            prob = prob.data.cpu()
        if i == 0:
            features = np.zeros((N, feat.shape[1]), dtype='float32')
            labels = torch.zeros(N, dtype=torch.long)
            probs = torch.zeros(N, 3)
        if i < len(dataloader) - 1:
            features[i * batch_size: (i + 1) * batch_size] = feat
            labels[i * batch_size: (i + 1) * batch_size] = batch[2]
            probs[i * batch_size: (i + 1) * batch_size] = prob
        else:
            # special treatment for final batch
            features[i * batch_size:] = feat
            labels[i * batch_size:] = batch[2]
            probs[i * batch_size:] = prob  # (N,classes)
    return features, labels, probs


def label_clean(soft_labels, features, labels, probs, temperature, n1, n2, k):
    # initalize knn search
    N = features.shape[0]
    index = faiss.IndexFlatIP(features.shape[1])

    index.add(features)
    D, I = index.search(features, k + 1)
    neighbors = torch.LongTensor(I)  # find k nearest neighbors excluding itself

    score = torch.zeros(N, 3)  # holds the score from weighted-knn
    weights = torch.exp(torch.Tensor(D[:, 1:]) / temperature)  # weight is calculated by embeddings' similarity
    for n in range(N):
        neighbor_labels = soft_labels[neighbors[n, 1:]]  # 每一个样本选出最近k个样本的p值 (k,classes)
        score[n] = (neighbor_labels * weights[n].unsqueeze(-1)).sum(0)  # aggregate soft labels from neighbors
    soft_labels = (score / score.sum(1).unsqueeze(
        -1) + probs) / 2  # combine with model's prediction as the new soft labels

    # consider the ground-truth label as clean if the soft label outputs a score higher than the threshold
    gt_score = soft_labels[labels >= 0, labels]
    clean_idx = gt_score > n1
    gt_noisy = ~(gt_score > n1)

    # get the hard pseudo label and the clean subset used to calculate supervised loss
    max_score, hard_labels = torch.max(soft_labels, 1)
    noisy_idx = (max_score > n2) & gt_noisy

    gt_hard = clean_idx | noisy_idx
    hard_idx = ~gt_hard
    print('Number of clean samples: %d' % clean_idx.sum())
    print('Number of hard samples: %d' % hard_idx.sum())
    print('Number of noisy samples: %d' % noisy_idx.sum())

    # save idx as np array
    # -----

    return soft_labels, clean_idx, hard_idx, noisy_idx


def get_classes(root: str):
    mri_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    mri_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(mri_class))
    return class_indices


def evaluate(confusion_matrix, pred, gt, n_classes):
    score_all_t = []
    confusion_matrix = torch.from_numpy(confusion_matrix)
    ACC = confusion_matrix.diag().sum() / confusion_matrix.sum()
    Pe = ((confusion_matrix.sum(0) / confusion_matrix.sum()) * (
            confusion_matrix.sum(1) / confusion_matrix.sum())).sum()
    kappa = 1 - (1 - ACC) / (1 - Pe + 1e-9)

    pred_probs = torch.softmax(pred.cpu(), dim=1).numpy()
    gt_bin = label_binarize(gt.cpu(), classes=[0, 1, 2])
    for i in range(n_classes):
        TP = confusion_matrix[i, i]
        FP = torch.sum(confusion_matrix[:, i]) - TP
        TN = confusion_matrix.diag().sum() - TP
        FN = confusion_matrix.sum() - TP - FP - TN
        # print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')

        SPE = TN / (TN + FP + 1e-10)
        SEN = TP / (TP + FN + 1e-10)
        PRE = TP / (TP + FP + 1e-10)
        F1 = 2 * PRE * SEN / (PRE + SEN)
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-9)
        # ROC computing
        fpr, tpr, th = roc_curve(gt_bin[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)

        score_all_t.append([SPE, SEN, PRE, F1, MCC, roc_auc])
    metric_mat = np.mean(np.array(score_all_t), axis=0)

    SPE, SEN, PRE, F1, MCC, roc_auc = metric_mat[0], metric_mat[1], metric_mat[2], metric_mat[3], metric_mat[4], \
        metric_mat[5]  # , metric_mat[6]
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
        if i == TN_ind:
            continue
        # import pdb; pdb.set_trace()
        fpr, tpr, th = roc_curve(gt_bin[:, i], pred_probs[:, i])  # pred_probs为正类的列
        roc_auc = auc(fpr, tpr)
        roc_auc_list.append(roc_auc)

    roc_auc_mean = np.mean(np.array(roc_auc_list), axis=0)

    return SPE, SEN, PRE, F1, MCC, roc_auc_mean


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    folds = [4, 3, 2, 1, 0]
    # folds = [0, 2, 3]
    csv_file = './evalute_all.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['Fold', 'val_acc', 'Epoch', 'train_acc', 'SPE', 'SEN', 'PRE', 'F1', 'Kappa', 'AUC', 'MCC', 'SPE_2',
             'SEN_2', 'PRE_2', 'F1_2', 'AUC_2', 'MCC_2'])
    for fold in folds:
        if os.path.exists(f"./weights{fold}") is False:
            os.makedirs(f"./weights{fold}")
        if os.path.exists(f"./runs/experiment{fold}") is False:
            os.makedirs(f"./runs/experiment{fold}")
        log_path = f"./runs/experiment{fold}"
        tb_writer = SummaryWriter(log_dir=log_path)
        data_path_train = args.train_data_path + f"/new_train_split{fold}.csv"
        data_path_val = args.val_data_path + f"/new_validation_split{fold}.csv"
        classes = get_classes(args.data_path)
        adni_dataloader = Adni_dataloader(data_path_train,
                                          data_path_val,
                                          transforms=True,
                                          double=True,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_workers)
        train_dataloader, val_dataloader, select_dataloader = adni_dataloader.run()

        model = resnet18().to(device)
        pg = [p for p in model.parameters() if p.requires_grad]
        learning_rate = args.lr
        optimizer = optim.Adam(pg, lr=learning_rate, weight_decay=args.weight_decay)  # momentum=args.momentum
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100], gamma=0.1)
        warmup_epochs = args.warmup_epochs
        epochs = args.epochs
        temperature = args.temperature
        n1 = args.n1
        n2 = args.n2
        w_pl_a = args.w_pl_a
        w_pl_m = args.w_pl_m
        w_uni = args.w_uni
        w_tri = args.w_tri
        k = args.k
        best_acc = 0.0

        # warm-up阶段
        for epoch in range(warmup_epochs):
            # train
            train_loss, train_acc, accuracy_dict, confusion_matrix_t, preds_roc_t, gts_roc_t = train_one_epoch_warmup(
                model=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                device=device,
                epoch=epoch,
                classes=classes,
            )  # gamma=gamma[epoch]

            # scheduler.step()
            # validate
            val_loss, val_acc, accuracy_dict, confusion_matrix_v, preds_roc_v, gts_roc_v = evaluate_warmup(
                model=model,
                data_loader=val_dataloader,
                device=device,
                epoch=epoch,
                classes=classes
            )  # gamma=gamma[epoch]

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "ad", "nc",
                    'mci']  # "test_loss", "test_acc"
            tags3 = ['SPE', 'SEN', 'PRE', 'F1', 'MCC', 'kappa', 'ROC']
            tags4 = ['SPE_2', 'SEN_2', 'PRE_2', 'F1_2', 'MCC_2', 'ROC_2']
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)

            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            tb_writer.add_scalar(tags[5], accuracy_dict[tags[5]], epoch)
            tb_writer.add_scalar(tags[6], accuracy_dict[tags[6]], epoch)
            tb_writer.add_scalar(tags[7], accuracy_dict[tags[7]], epoch)

            ACC, SPE, SEN, PRE, F1, MCC, kappa, ROC = evaluate(confusion_matrix_v, preds_roc_v, gts_roc_v, 3)
            SPE_2, SEN_2, PRE_2, F1_2, MCC_2, ROC_2 = evaluate_2(confusion_matrix_v, 2, preds_roc_v, gts_roc_v, 3)
            del preds_roc_t, gts_roc_t, preds_roc_v, gts_roc_v

            tb_writer.add_scalar(tags3[0], SPE, epoch)
            tb_writer.add_scalar(tags3[1], SEN, epoch)
            tb_writer.add_scalar(tags3[2], PRE, epoch)
            tb_writer.add_scalar(tags3[3], F1, epoch)
            tb_writer.add_scalar(tags3[4], MCC, epoch)
            tb_writer.add_scalar(tags3[5], kappa, epoch)
            tb_writer.add_scalar(tags3[6], ROC, epoch)

            tb_writer.add_scalar(tags4[0], SPE_2, epoch)
            tb_writer.add_scalar(tags4[1], SEN_2, epoch)
            tb_writer.add_scalar(tags4[2], PRE_2, epoch)
            tb_writer.add_scalar(tags4[3], F1_2, epoch)
            tb_writer.add_scalar(tags4[4], MCC_2, epoch)
            tb_writer.add_scalar(tags4[5], ROC_2, epoch)
            del train_loss, train_acc, confusion_matrix_t, val_loss, val_acc, confusion_matrix_v
            del ACC, SPE, SEN, PRE, F1, MCC, kappa, ROC, SPE_2, SEN_2, PRE_2, F1_2, MCC_2, ROC_2

        torch.cuda.empty_cache()

        for epoch in range(warmup_epochs, warmup_epochs + epochs):

            features, labels, probs = compute_features(select_dataloader, model, adni_dataloader.len_train_dataset,
                                                       device)
            if epoch == warmup_epochs:
                soft_labels = probs.clone()
            soft_labels, clean_idx, hard_idx, noisy_idx = label_clean(soft_labels, features, labels, probs, temperature,
                                                                      n1, n2, k)

            train_loss, train_acc, ce_t_loss, accuracy_dict, pl_a, pl_m, uni, confusion_matrix_t, preds_list_t, gts_list_t = train_one_epoch_select(
                model=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                device=device,
                epoch=epoch,
                classes=classes,
                clean_idx=clean_idx,
                hard_idx=hard_idx,
                noisy_idx=noisy_idx,
                temperature=temperature,
                w_pl_a=w_pl_a,
                w_pl_m=w_pl_m,
                w_uni=w_uni,
                w_tri=w_tri)

            scheduler.step()
            # validate
            val_loss, val_acc, accuracy_dict, confusion_matrix_v, preds_list_v, gts_list_v = evaluate_listwise(
                model=model,
                data_loader=val_dataloader,
                device=device,
                epoch=epoch,
                classes=classes
            )

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "ad", "nc",
                    'mci', 'ce_loss', 'pl_a', 'pl_m', 'scl', 'tri', 'uni']  # "test_loss", "test_acc"
            tags2 = ['easy_percent', 'hard_percent', 'noisy_percent']
            tags3 = ['SPE', 'SEN', 'PRE', 'F1', 'MCC', 'kappa', 'ROC']
            tags4 = ['SPE_2', 'SEN_2', 'PRE_2', 'F1_2', 'MCC_2', 'ROC_2']
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            # tb_writer.add_scalar(tags[8], test_loss, epoch)
            # tb_writer.add_scalar(tags[9], test_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            tb_writer.add_scalar(tags[5], accuracy_dict[tags[5]], epoch)
            tb_writer.add_scalar(tags[6], accuracy_dict[tags[6]], epoch)
            tb_writer.add_scalar(tags[7], accuracy_dict[tags[7]], epoch)
            tb_writer.add_scalar(tags[8], float(ce_t_loss), epoch)
            tb_writer.add_scalar(tags[9], float(pl_a), epoch)
            tb_writer.add_scalar(tags[10], float(pl_m), epoch)
            # tb_writer.add_scalar(tags[11], float(scl), epoch)
            # tb_writer.add_scalar(tags[12], float(tri), epoch)
            tb_writer.add_scalar(tags[13], float(uni), epoch)

            all_idx_sum = clean_idx.sum() + hard_idx.sum() + noisy_idx.sum()
            tb_writer.add_scalar(tags2[0], clean_idx.sum() / (all_idx_sum + 1e-10), epoch)
            tb_writer.add_scalar(tags2[1], hard_idx.sum() / (all_idx_sum + 1e-10), epoch)
            tb_writer.add_scalar(tags2[2], noisy_idx.sum() / (all_idx_sum + 1e-10), epoch)

            preds_roc_v = torch.cat(preds_list_v, dim=0)
            gts_roc_v = torch.cat(gts_list_v, dim=0)
            ACC, SPE, SEN, PRE, F1, MCC, kappa, ROC = evaluate(confusion_matrix_v, preds_roc_v, gts_roc_v, 3)
            SPE_2, SEN_2, PRE_2, F1_2, MCC_2, ROC_2 = evaluate_2(confusion_matrix_v, 2, preds_roc_v, gts_roc_v, 3)
            del preds_roc_v, gts_roc_v

            tb_writer.add_scalar(tags3[0], SPE, epoch)
            tb_writer.add_scalar(tags3[1], SEN, epoch)
            tb_writer.add_scalar(tags3[2], PRE, epoch)
            tb_writer.add_scalar(tags3[3], F1, epoch)
            tb_writer.add_scalar(tags3[4], MCC, epoch)
            tb_writer.add_scalar(tags3[5], kappa, epoch)
            tb_writer.add_scalar(tags3[6], ROC, epoch)

            tb_writer.add_scalar(tags4[0], SPE_2, epoch)
            tb_writer.add_scalar(tags4[1], SEN_2, epoch)
            tb_writer.add_scalar(tags4[2], PRE_2, epoch)
            tb_writer.add_scalar(tags4[3], F1_2, epoch)
            tb_writer.add_scalar(tags4[4], MCC_2, epoch)
            tb_writer.add_scalar(tags4[5], ROC_2, epoch)

            if val_acc > best_acc or val_acc >= 0.75:
                best_acc = val_acc
                print('best_acc:', best_acc)
                # logging.info("Best_acc:" + str(best_acc))
                if val_acc >= 0.72:
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [fold, val_acc, epoch, train_acc, SPE, SEN, PRE, F1, kappa, ROC, MCC,
                             SPE_2, SEN_2, PRE_2, F1_2, ROC_2, MCC_2])
                    torch.save(model.state_dict(),
                               f"./weights{fold}/{fold}-{round(val_acc, 3)}-{epoch}-{round(train_acc, 3)}.pth")
                    print('Best Model saved.')

            del train_loss, train_acc, confusion_matrix_t, val_loss, val_acc, confusion_matrix_v
            del ACC, SPE, SEN, PRE, F1, MCC, kappa, ROC, SPE_2, SEN_2, PRE_2, F1_2, MCC_2, ROC_2
        print('finished!')
    stop = datetime.now()
    print("Running time: ", stop - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('-warmup_epochs', type=int, default=20)  # 25
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--start_epochs', type=int, default=84)
    parser.add_argument('--batch_size', type=int, default=10)  # 16
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=0)  # 4

    # 载入数据相关

    parser.add_argument('--data_path', type=str,
                        default=r"D:\DJC\dataset")  # 改
    parser.add_argument('--weights1', type=str, default='',  # 经过多个数据的预训练 ./resnet_18_23dataset.pth
                        help='initial weights path')

    # 复现相关
    parser.add_argument('--val_data_path', type=str,
                        default=r"D:\DJC\Ablation\ADNI_data_notest_age\validation_n")  # 改
    parser.add_argument('--train_data_path', type=str,
                        default=r"D:\DJC\Ablation\ADNI_data_notest_age\train_n")  # 改
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # 5e-2
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--momentum', type=float, default=0.9)
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')  # cuda:1
    parser.add_argument('--tensor_root', default='./weights/', help='tensorboard save path')

    # model
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--w_pl_m', type=float, default=0.125)
    parser.add_argument('--w_pl_a', type=float, default=0.025)
    parser.add_argument('--w_tri', type=float, default=0.05)
    parser.add_argument('--w_uni', type=float, default=1.0)
    parser.add_argument('--n1', type=float, default=0.7)
    parser.add_argument('--n2', type=float, default=0.6)
    parser.add_argument('--k', type=int, default=16)
    opt = parser.parse_args()

    main(opt)
