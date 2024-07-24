#主要是做病理图像模型的五折交叉验证和全训练集训练
import os
import pandas as pd
import torch
import copy
import json
import warnings
import torch.nn as nn
import numpy as np
from train import train_epoch,test
from torch.utils.data import DataLoader
import torchvision.models as models
from model import Resnet18,Resnet50,Swin_Transforemr,Densenet121,CNN_SelfAttention_Parallel_concat,CNN_SelfAttention_softmax,CNN_SelfAttention_attn
from torch.optim import lr_scheduler
from dataset import My_Dataset
from utils import draw_LossAcc
import random

def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main_cross(patch_path,crossval_path,train_bs,test_bs,lr,out_num,epochs,output_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(crossval_path,'r') as f:
        data_index = json.load(f)

    test_index = data_index[str(0)][2]
    test_dataset = My_Dataset(patch_path,test_index,'test')
    test_loader = DataLoader(test_dataset,batch_size=test_bs,shuffle=False)

    cross_val_result = pd.DataFrame()
    for fold in range(5):

        if not os.path.exists(os.path.join(output_path,str(fold))):
            os.makedirs(os.path.join(output_path,str(fold)))
        save_path_fold = os.path.join(output_path,str(fold))

        #定义数据集======用训练集+验证机来做训练集，仅做一次测试=============
        train_index,val_index = data_index[str(fold)][0],data_index[str(fold)][1]
        train_dataset = My_Dataset(patch_path,train_index,'train')
        val_dataset = My_Dataset(patch_path,val_index,'test')
        train_loader = DataLoader(train_dataset,batch_size=train_bs,shuffle=True,drop_last=True)
        val_loader = DataLoader(val_dataset,batch_size=test_bs,shuffle=False,drop_last=False)
        #
        #=========RESTART==================
        # 定义数据集
        train_index,val_index = [i for i in data_index[str(fold)][0] if len(i)>7],[j for j in data_index[str(fold)][1] if len(j)>7]
        train_dataset = My_Dataset(patch_path,train_index,'train')
        val_dataset = My_Dataset(patch_path,val_index,'test')
        train_loader = DataLoader(train_dataset,batch_size=train_bs,shuffle=True,drop_last=True)
        val_loader = DataLoader(val_dataset,batch_size=test_bs,shuffle=False,drop_last=False)

        #定义模型
        net = CNN_SelfAttention_softmax('resnet18',out_num)
        net.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99))
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


        #训练过程
        train_loss = []  # 用来画图
        train_acc = []  # 用来画图
        loss_best = 100
        stop_index = 0
        threshold_best = 0
        for epoch in range(epochs):
            net, loss, accuracy, threshold = train_epoch(net, train_loader, criterion, optimizer, device)
            print(f'{epoch}: {loss}')
            exp_lr_scheduler.step()
            train_loss.append(loss)
            train_acc.append(accuracy)
            if loss < loss_best:
                loss_best = loss
                threshold_best = threshold
                best_model_wts = copy.deepcopy(net.state_dict())
                stop_index = 0
            else:
                stop_index += 1
            if stop_index >= 5:
                break

        draw_LossAcc(epoch, 'train', train_loss, train_acc, save_path_fold)
        model_save_path = os.path.join(save_path_fold, 'Model')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(best_model_wts, os.path.join(model_save_path, 'best.pkl'))

        net = CNN_SelfAttention_softmax('resnet18', out_num)
        net.load_state_dict(best_model_wts)
        net.to(device)
        # patchroc, patchpr, patchacc, patchpre, patchrecall, patchf1 = test(
        #     net, val_loader, criterion, 0.5, device, save_path_fold)
        patchroc,patchpr,patchacc,patchpre,patchrecall,patchf1,wsiroc,wsipr,wsiacc,wsipre,wsirecall,wsif1 = test(net, val_loader, criterion, 0.5, device, save_path_fold)
        cross_val_result.loc[str(fold),'fold'] = fold
        cross_val_result.loc[str(fold),'patch_roc'] = patchroc
        cross_val_result.loc[str(fold), 'patch_pr'] = patchpr
        cross_val_result.loc[str(fold), 'patch_accuracy'] = patchacc
        cross_val_result.loc[str(fold), 'patch_precision'] = patchpre
        cross_val_result.loc[str(fold), 'patch_recall'] = patchrecall
        cross_val_result.loc[str(fold), 'patch_f1'] = patchf1

        cross_val_result.loc[str(fold), 'wsi_roc'] = wsiroc
        cross_val_result.loc[str(fold), 'wsi_pr'] = wsipr
        cross_val_result.loc[str(fold), 'wsi_accuracy'] = wsiacc
        cross_val_result.loc[str(fold), 'wsi_precision'] = wsipre
        cross_val_result.loc[str(fold), 'wsi_recall'] = wsirecall
        cross_val_result.loc[str(fold), 'wsi_f1'] = wsif1
    cross_val_result.to_csv(os.path.join(output_path,'CrossVal_result.csv'))



def main(patch_path,crossval_path,train_bs,test_bs,lr,out_num,epochs,output_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(crossval_path,'r') as f:
        data_index = json.load(f)

    test_index = data_index[str(0)][2]
    train_index, val_index = data_index[str(0)][0], data_index[str(0)][1]

    test_dataset = My_Dataset(patch_path,test_index,'test')
    test_loader = DataLoader(test_dataset,batch_size=test_bs,shuffle=False,drop_last=False)

    train_dataset = My_Dataset(patch_path, train_index+val_index, 'train')
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True,drop_last=True)


    # #定义模型
    net = CNN_SelfAttention_softmax('resnet18',out_num)
    net.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    
    #训练过程
    train_loss = []  # 用来画图
    train_acc = []  # 用来画图
    loss_best = 100
    stop_index = 0
    threshold_best = 0
    for epoch in range(epochs):
        net, loss, accuracy, threshold = train_epoch(net, train_loader, criterion, optimizer, device)
        print(f'{epoch}: {loss}')
        exp_lr_scheduler.step()
        train_loss.append(loss)
        train_acc.append(accuracy)
        if loss < loss_best:
            loss_best = loss
            threshold_best = threshold
            best_model_wts = copy.deepcopy(net.state_dict())
            stop_index = 0
        else:
            stop_index += 1
        if stop_index >= 5:
            break
    
    draw_LossAcc(epoch, 'train', train_loss, train_acc, output_path)
    model_save_path = os.path.join(output_path, 'Model')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(best_model_wts, os.path.join(model_save_path, 'best.pkl'))

    best_model_wts = torch.load('./0.5Scale-Results/Model/best.pkl')
    # net = Resnet18(2)
    # net = Swin_Transforemr(model_size='b',pretrained_dataset='1k',out_num = out_num)
    net = CNN_SelfAttention_softmax('resnet18', out_num)
    net.load_state_dict(best_model_wts)
    net.to(device)
    test(net, test_loader, criterion, 0.5, device, output_path)

def main_contrast(patch_path,train_index_path,test_index_path,train_bs,test_bs,lr,out_num,epochs,output_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(train_index_path,'r') as f:
        train_index = json.load(f)

    train_index, val_index = train_index[str(0)][0], train_index[str(0)][1]
    indexs = train_index+val_index
    indexs_none = [i for i in indexs if len(i)>7]
    train_dataset = My_Dataset(patch_path, indexs_none, 'train')
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True,drop_last=True)

    test_index_list = list(pd.read_csv(test_index_path)['ID'])
    test_index = [i for j in test_index_list for i in j.split(',') if len(i)>7]
    test_dataset = My_Dataset(patch_path,test_index,'test')
    test_loader = DataLoader(test_dataset,batch_size=test_bs,shuffle=False,drop_last=True)


    #定义模型
    net = CNN_SelfAttention_softmax('resnet18',out_num)
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    #训练过程
    train_loss = []  # 用来画图
    train_acc = []  # 用来画图
    loss_best = 100
    stop_index = 0
    threshold_best = 0
    for epoch in range(epochs):
        net, loss, accuracy, threshold = train_epoch(net, train_loader, criterion, optimizer, device)
        print(f'{epoch}: {loss}')
        exp_lr_scheduler.step()
        train_loss.append(loss)
        train_acc.append(accuracy)
        if loss < loss_best:
            loss_best = loss
            threshold_best = threshold
            best_model_wts = copy.deepcopy(net.state_dict())
            stop_index = 0
        else:
            stop_index += 1
        if stop_index >= 5:
            break

    draw_LossAcc(epoch, 'train', train_loss, train_acc, output_path)
    model_save_path = os.path.join(output_path, 'Model')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(best_model_wts, os.path.join(model_save_path, 'best.pkl'))

    best_model_wts = torch.load('/2data/liyixin/HCC/data/MultiFactor_prediction/Results/13-resnet18-selfattn-softmax/No_PathoModel_Train_Contrast/Model/best.pkl')
    net = Resnet18(2)
    net = Swin_Transforemr(model_size='b',pretrained_dataset='1k',out_num = out_num)
    net = CNN_SelfAttention_softmax('resnet18', out_num)
    net.load_state_dict(best_model_wts)
    net.to(device)
    test(net, test_loader, criterion, 0.5, device, output_path)

def save_diffScale_crossesult_sameNumas25(cross_path,patch_path,diff_result_path,batch_size,device,output_path):
    lst = ['0.33','0.5','0.66','0.75','all']
    for ratio in lst:
        cross_val_result = pd.DataFrame()
        single_cross_path = os.path.join(cross_path, ratio + 'crossVal.json')
        cross_result_savepath = os.path.join(output_path,ratio+'Scale-Results')
        with open(single_cross_path, 'r') as f:
            data_index = json.load(f)
        for fold in range(5):
            save_path = os.path.join(output_path,ratio+'Scale-Results',str(fold),'sameNum_as25_results')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            val_index = list(np.random.choice(data_index[str(fold)][1],22,replace=False))
            val_dataset = My_Dataset(patch_path,val_index,'test')
            val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
            wts_path = os.path.join(diff_result_path,ratio+'Scale-Results',str(fold),'Model','best.pkl')
            net = Resnet18(2)
            net.load_state_dict(torch.load(wts_path))
            net.to(device)
            criterion = nn.CrossEntropyLoss()
            patchroc,patchpr,patchacc,patchpre,patchrecall,patchf1,wsiroc,wsipr,wsiacc,wsipre,wsirecall,wsif1 = test(net, val_loader, criterion, 0.5, device, save_path)
            cross_val_result.loc[str(fold), 'fold'] = fold
            cross_val_result.loc[str(fold), 'patch_roc'] = patchroc
            cross_val_result.loc[str(fold), 'patch_pr'] = patchpr
            cross_val_result.loc[str(fold), 'patch_accuracy'] = patchacc
            cross_val_result.loc[str(fold), 'patch_precision'] = patchpre
            cross_val_result.loc[str(fold), 'patch_recall'] = patchrecall
            cross_val_result.loc[str(fold), 'patch_f1'] = patchf1

            cross_val_result.loc[str(fold), 'wsi_roc'] = wsiroc
            cross_val_result.loc[str(fold), 'wsi_pr'] = wsipr
            cross_val_result.loc[str(fold), 'wsi_accuracy'] = wsiacc
            cross_val_result.loc[str(fold), 'wsi_precision'] = wsipre
            cross_val_result.loc[str(fold), 'wsi_recall'] = wsirecall
            cross_val_result.loc[str(fold), 'wsi_f1'] = wsif1
        cross_val_result.to_csv(os.path.join(cross_result_savepath, 'CrossVal_result_sameNum_as25.csv'))


if __name__ == '__main__':
    seed_torch()
    warnings.filterwarnings('ignore')
    patch_path = './New_All_ColorNormalization_sorafenib_vahadane/'
    # patch_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/ALL_ColorNormalization_patch/'
    # scale = ['0.25','0.33','0.5','0.66','0.75','all']
    scale = ['0.25','0.5','0.75', 'all']
    # scale = ['0.5']
    #=====0416=====直接将所有data_index传进去=========
    label = pd.read_excel('./integrate_dataset_clinical_feature.xlsx')
    # all_index =  [j for i in label['ID'] for j in i.split(',')]
    for idx in range(len(scale)):
        single_scale = scale[idx]
        crossval_path = './'+single_scale+'crossVal.json'
        output_path = os.path.join('./01-CNN-SASM/',single_scale+'Scale-Results')
        makedir(output_path)
        # main(patch_path,crossval_path,128,128,0.0001,2,50,output_path)
        main_cross(patch_path, crossval_path, 128, 128, 0.0001, 2, 50, output_path)

