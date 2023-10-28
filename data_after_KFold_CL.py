import random
import time
import json
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from FocalLoss import FocalLoss
from dataloader import dialogDataset
from model_1 import BugListener
from cleanlab.filter import find_label_issues
from sklearn.model_selection import KFold


seed = 2021


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def data_write(file_path, data):
    from openpyxl import load_workbook
    wb = load_workbook(filename=file_path)
    wsw = wb.active
    wsw.append(data)
    wb.save(filename=file_path)

def get_data_dict(train_address, test_address):
    with open(train_address + '_train.json', 'r') as r:
        train_data_dict = json.load(r)

    with open(test_address + '_test.json', 'r') as r:
        test_data_dict = json.load(r)
    return train_data_dict, test_data_dict


def get_dialog_loaders(train_data_dict, test_data_dict, pretrained_model, batch_size=32, num_workers=0, pin_memory=False):

    train_set = dialogDataset(train_data_dict, pretrained_model)

    test_set = dialogDataset(test_data_dict, pretrained_model)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=train_set.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             collate_fn=test_set.collate_fn,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader


def train_model(model, loss_func, dataloader, optimizer, epoch, cuda):
    losses, preds, labels, ids = [], [], [], []

    model.train()

    seed_everything(seed + epoch)
    for data in tqdm(dataloader):
        # clear the grad
        optimizer.zero_grad()

        input_ids, token_type_ids, attention_mask_ids, graph_label = \
            [d.cuda() for d in data[:-4]] if cuda else data[:-4]
        dialog_id, role, graph_edge, seq_len = data[-4:]

        log_prob = model(input_ids, token_type_ids, attention_mask_ids, role, seq_len, graph_edge)
        loss = loss_func(log_prob, graph_label)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(graph_label.cpu().numpy())
        losses.append(loss.item())
        ids += dialog_id

        # accumulate the grad
        loss.backward()
        # optimizer the parameters
        optimizer.step()
        # print("preds=",preds)

    if preds:
        # print("predsss=",preds)
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

    assert len(preds) == len(labels) == len(ids)
    error_ids = []
    for vid, target, pred in zip(ids, labels, preds):
        if target != pred:
            error_ids.append(vid)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    graph_pre = round(precision_score(labels, preds) * 100, 2)
    graph_rec = round(recall_score(labels, preds) * 100, 2)
    graph_fscore = round(f1_score(labels, preds) * 100, 2)

    return avg_loss, graph_pre, graph_rec, graph_fscore, error_ids


def evaluate_model(model, loss_func, dataloader, cuda):
    losses, preds, labels, ids, pred_probs = [], [], [], [], []

    model.eval()

    seed_everything(seed)
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_ids, token_type_ids, attention_mask_ids, graph_label = \
                [d.cuda() for d in data[:-4]] if cuda else data[:-4]
            dialog_id, role, graph_edge, seq_len = data[-4:]

            log_prob = model(input_ids, token_type_ids, attention_mask_ids, role, seq_len, graph_edge)
            loss = loss_func(log_prob, graph_label)

            prob = torch.exp(log_prob)
            pred_probs.append(prob.cpu().numpy())

            preds.append(torch.argmax(log_prob, 1).cpu().numpy())
            labels.append(graph_label.cpu().numpy())
            losses.append(loss.item())
            ids += dialog_id

    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        pred_probs = np.concatenate(pred_probs)

    assert len(preds) == len(labels) == len(ids)
    error_ids = []
    for vid, target, pred in zip(ids, labels, preds):
        if target != pred:
            error_ids.append(vid)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    graph_pre = round(precision_score(labels, preds) * 100, 2)
    graph_rec = round(recall_score(labels, preds) * 100, 2)
    graph_fscore = round(f1_score(labels, preds) * 100, 2)

    return avg_loss, graph_pre, graph_rec, graph_fscore, error_ids, pred_probs, labels, ids


class Config(object):
    def __init__(self, project):
        self.cuda = True
        self.train_address = f"./data/processed/{project}"  # angular, appium, docker, dl4j, gitter, typescript
        self.test_address = f"./data/processed/{project}"
        self.pretrained_model = './../pretrained_model/bert_base_uncased'
        self.D_bert = 768
        self.filter_sizes = [2, 3, 4, 5]
        self.filter_num = 50
        self.D_cnn = 100
        self.D_graph = 64
        self.lr = 1e-4
        self.l2 = 1e-5
        self.batch_size = 8
        self.graph_class_num = 2
        self.dropout = 0.5
        self.epochs = 30



def dict_to_numpy(data_dict):
    ids = data_dict['ids']
    dialog = data_dict['dialog']
    role = data_dict['role']
    label = data_dict['label']
    edge = data_dict['edge']
    data = []
    for i in range(len(ids)):
        lst = []
        lst.append(ids[i])
        lst.append(dialog[i])
        lst.append(role[i])
        lst.append(label[i])
        lst.append(edge[i])
        data.append(lst)

    data = np.array(data, dtype=object)

    return data

def numpy_to_dict(data):
    data_list = data.tolist()
    data_dict = {}
    ids, dialog, role, label, edge = [], [], [], [], []
    for i in range(len(data_list)):
        ids.append(data_list[i][0])
        dialog.append(data_list[i][1])
        role.append(data_list[i][2])
        label.append(data_list[i][3])
        edge.append(data_list[i][4])

    data_dict['ids'] = ids
    data_dict['dialog'] = dialog
    data_dict['role'] = role
    data_dict['label'] = label
    data_dict['edge'] = edge

    return data_dict




def K_fold(project, data_dict, config, k_folds=10, epochs=30, ifcuda=True):
    data = dict_to_numpy(data_dict)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    total_err_ids = []
    ids_all = []

    for train_idx, valid_idx in kf.split(data):
        # print('train_index:%s , test_index: %s ' % (train_idx, valid_idx))
        # print("train_idx_len={}, test_idx_len={}".format(len(train_idx), len(valid_idx)))
        train_data = data[train_idx]
        valid_data = data[valid_idx]
        lst = []
        for j in valid_data:
            lst.append(j[0])
        ids_all.append(lst)

        continue

    for i in range(k_folds):
        for j in range(i+1, k_folds):
            for k in ids_all[i]:
                if k in ids_all[j]:
                    print("出现了重复的样本！！！")
    print("SSSSSSSSSSS")




    #     train_data_dict = numpy_to_dict(train_data)
    #     valid_data_dict = numpy_to_dict(valid_data)
    #
    #
    #     train_loader, dev_loader = get_dialog_loaders(train_data_dict, valid_data_dict, pretrained_model=config.pretrained_model,
    #                                                    batch_size=config.batch_size)
    #     model = BugListener(config.pretrained_model, config.D_bert, config.filter_sizes, config.filter_num,
    #                         config.D_cnn, config.D_graph, n_speakers=2, graph_class_num=config.graph_class_num,
    #                         dropout=config.dropout, ifcuda=cuda)
    #
    #     graph_loss = FocalLoss(gamma=2)
    #
    #     for name, params in model.pretrained_bert.named_parameters():
    #         params.requires_grad = False
    #     print(get_parameter_number(model))
    #
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr,
    #                            weight_decay=config.l2)
    #     if cuda:
    #         model.cuda()
    #     best_f1 = 0
    #     PATH = 'best_' + str(project) + '_model.pth'
    #     for epoch in range(0):
    #         train_loss, train_pre, train_rec, train_f1, train_error = \
    #             train_model(model, graph_loss, train_loader, optimizer, epoch, ifcuda)
    #
    #         test_loss, test_pre, test_rec, test_f1, test_error, _, _, _ = \
    #             evaluate_model(model, graph_loss, dev_loader, cuda)
    #
    #         print('epoch:{},train_loss:{},train_pre:{},train_rec:{},train_f1:{},test_loss:{},test_graph_pre:{},'
    #               'test_graph_rec:{},test_graph_f1:{}'.format(epoch + 1, train_loss, train_pre, train_rec, train_f1,
    #                                                                      test_loss, test_pre, test_rec, test_f1))
    #         if test_f1 > best_f1:
    #             best_f1 = test_f1
    #             torch.save(model.state_dict(), PATH)
    #
    #     # model.load_state_dict(torch.load(PATH))
    #     _, _, _, _, _, pred_probs, labels, ids = evaluate_model(model, graph_loss, dev_loader, cuda)
    #
    #     ranked_label_issues = find_label_issues(
    #         labels=labels,
    #         pred_probs=pred_probs,
    #         return_indices_ranked_by="self_confidence",
    #     )
    #     name = []
    #     for i in ranked_label_issues:
    #         name.append(ids[i])
    #         total_err_ids.append(ids[i])
    #     print("当前的错误标签ids为： ",name)
    # return total_err_ids

if __name__ == '__main__':
    for project in ['angular', 'appium', 'docker', 'dl4j', 'gitter', 'typescript']:
        print("======current_project:", project, "======")
        config = Config(project=project)
        cuda = torch.cuda.is_available() and config.cuda
        if cuda:
            print('Running on GPU')
        else:
            print('Running on CPU')

        seed_everything(seed)

        train_data_dict, test_data_dict = get_data_dict(config.train_address, config.test_address)

        total_err_ids = K_fold(project, train_data_dict, k_folds=10, epochs=10, config=config, ifcuda=cuda)
        print("{}项目共去掉了噪音样本{}个！！".format(project, len(total_err_ids)))

        filename_path = "err_ids.xlsx"
        output = [str(project)] + total_err_ids
        data_write(filename_path, output)
        datafile_path = config.train_address+"_CL_train.json"
        ids, dialog, role, edge, label = [], [], [], [], []
        new_train_dict = {}
        for i in range(len(train_data_dict['ids'])):
            id = train_data_dict['ids'][i]
            if id not in total_err_ids:
                ids.append(train_data_dict['ids'][i])
                dialog.append(train_data_dict['dialog'][i])
                role.append(train_data_dict['role'][i])
                label.append(train_data_dict['label'][i])
                edge.append(train_data_dict['edge'][i])
        new_train_dict['ids'] = ids
        new_train_dict['dialog'] = dialog
        new_train_dict['role'] = role
        new_train_dict['label'] = label
        new_train_dict['edge'] = edge

        with open(datafile_path, encoding='utf-8', mode='w') as f:
            f.write(json.dumps(new_train_dict))








