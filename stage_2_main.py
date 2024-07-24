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
from model import Resnet18,MultiFactor_Model,Fusion_model,Simple_Concat,Simple_Concat_oneModality,Pathological_predict_prognosis
from torch.optim import lr_scheduler
from dataset import WSI_MultiFactor_Dataset,TCGA_MultiFactor_Dataset
from utils import draw_LossAcc,auroc_pr,draw_confusion_matrix
import random
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score,precision_recall_curve,average_precision_score,auc,roc_curve,precision_score,recall_score,f1_score,roc_auc_score


def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class WSI_Pathology_Feature():
    def __init__(self,model,whole_wsi_path,save_path,device):
        self.model = model
        self.save_path = save_path
        self.path = whole_wsi_path
        self.device = device
        self.test_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.wsi_list = self.get_wsi_list()
        # self.wsi_feature = self.get_wsi_pathological_feature()

    def get_wsi_list(self):
        self.wsi_list = []
        for slide in os.listdir(self.path):
            slide_path = os.path.join(self.path,slide)
            self.wsi_list.append(slide_path)
        return self.wsi_list

    def save_patch_pathological_feature(self):
        self.model.eval()
        patch_feature = {}
        with torch.no_grad():
            for wsi in self.wsi_list:
                for patch in os.listdir(wsi):
                    patch_path = os.path.join(wsi,patch)
                    patch_preprocess = self.patch_preprocess(patch_path)
                    feature, _ = self.model(patch_preprocess.unsqueeze(0).to(self.device))
                    patch_feature[patch.split('.')[0]] = feature.squeeze().detach().cpu().numpy().tolist()
        with open(os.path.join(self.save_path,'patch_pathological_feature.json'),'w') as f:
            json.dump(patch_feature,f)


    def get_wsi_pathological_feature(self):
        self.model.eval()
        wsi_feature = {}
        with torch.no_grad():
            for wsi in self.wsi_list:
                single_wsi_feature = []
                for patch in os.listdir(wsi):
                    patch_preprocess = self.patch_preprocess(os.path.join(wsi,patch))
                    feature,_ = self.model(patch_preprocess.unsqueeze(0).to(self.device))
                    single_wsi_feature.append(feature.squeeze().detach().cpu().numpy())
                single_wsi_feature = np.mean(single_wsi_feature,axis=0)
                wsi_feature[os.path.basename(wsi)] = single_wsi_feature.tolist()
        return wsi_feature

    def save_feature(self,wsi_feature):
        with open(os.path.join(self.save_path,'wsi_pathological_feature.json'),'w') as file:
            json.dump(wsi_feature,file)

    def patch_preprocess(self,path):
        '''
        将每张patch转化为tensor并归一化
        :param path:
        :return:
        '''
        image = Image.open(path)
        return self.test_process(image)

def save_wsi_pathological_feature(tile_path,save_path,model_wts_path,device):
    model = Resnet18(2)
    model.load_state_dict(torch.load(model_wts_path))
    model.to(device)
    generator = WSI_Pathology_Feature(model, tile_path, save_path,device)
    wsi_feature = generator.get_wsi_pathological_feature()
    generator.save_feature(wsi_feature)

def save_patch_pathological_feature(tile_path,save_path,model_wts_path,device):
    model = Resnet18(2)
    model.load_state_dict(torch.load(model_wts_path))
    model.to(device)
    generator = WSI_Pathology_Feature(model, tile_path, save_path,device)
    generator.save_patch_pathological_feature()

class MultiFactor_main_WSI_level():
    def __init__(self,integrate_path,pathological_path,split_path,train_bs,test_bs,lr,epoch,output_path):
        self.data_path = integrate_path
        self.patho_fea_path = pathological_path
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.in_num = 512
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = epoch
        with open(split_path,'r') as f:
            self.split_data = json.load(f)
        self.lr = lr
        self.output_path = output_path

    def get_data(self,data_index,mode,tcga=False,reverse = False):
        if not tcga:
            dataset = WSI_MultiFactor_Dataset(self.patho_fea_path,self.data_path,data_index,reverse)
        else:
            dataset = TCGA_MultiFactor_Dataset()

        if mode == 'train':
            data_loader = DataLoader(dataset,batch_size=self.train_bs,shuffle=True)
        # elif mode == 'test':
        #     data_loader = DataLoader(dataset,batch_size=self.test_bs,shuffle=False)
        else:
            data_loader = DataLoader(dataset, batch_size=self.test_bs, shuffle=False)
        return data_loader

    def get_model(self):
        # config = [[32, 32, 32], [64, 64, 64], [128, 128, 256]]
        # config = [[8,8],[16,16],[64,128]]
        # config = [[4,4], [8,8], [32, 128]]
        # model = Fusion_model(128,config,2)
        # model = MultiFactor_Model(128)
        # model = Simple_Concat(128)
        # model = Simple_Concat_oneModality(128)
        model = Pathological_predict_prognosis(128)
        model.to(self.device)
        return model


    def cross_train(self):
        cross_result = pd.DataFrame()
        for fold in range(5):
            output_path = os.path.join(self.output_path,str(fold))
            self.makefile(output_path)
            train_index = [i for i in self.split_data[str(fold)][0] if len(i)>7]
            val_index = [i for i in self.split_data[str(fold)][1] if len(i)>7]

            train_loader = self.get_data(train_index,'train')
            val_loader = self.get_data(val_index,'test')

            net = self.get_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, betas=(0.9, 0.99))
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            train_loss = []
            train_acc = []
            loss_best = 100
            stop_index = 0
            for epoch in range(self.epochs):
                net, loss, accuracy = self.train_epoch(net, train_loader, criterion, optimizer)
                print(f'{epoch} {loss}')
                exp_lr_scheduler.step()
                train_loss.append(loss)
                train_acc.append(accuracy)
                if loss < loss_best:
                    loss_best = loss
                    best_model_wts = copy.deepcopy(net.state_dict())
                    stop_index = 0
                else:
                    stop_index += 1
                # if stop_index >= 5:
                #     break
            draw_LossAcc(epoch, 'train', train_loss, train_acc, output_path)
            model_save_path = os.path.join(output_path, 'Model')
            self.makefile(model_save_path)
            torch.save(best_model_wts, os.path.join(model_save_path, 'best.pkl'))

            net = self.get_model()
            net.load_state_dict(best_model_wts)
            net.eval()
            label_list = []
            score_list = []
            prediction_list = []
            patients_list = []
            feature_dict = {}
            with torch.no_grad():
                for idx,sample in enumerate(val_loader):
                    patho_fea, clinical_fea, therapy_fea = sample['patho_feature'], sample['clinical_feature'], sample['therapy_feature']
                    label, patient_id = sample['label'], sample['patient_id']
                    patho_fea = patho_fea.to(self.device)
                    clinical_fea = clinical_fea.to(self.device)
                    therapy_fea = therapy_fea.to(self.device)
                    output = net(patho_fea)
                    score = output[:, 1]
                    prediction = torch.argmax(output, dim=-1)

                    score_list += score.detach().cpu().numpy().tolist()
                    label_list += label.detach().cpu().numpy().tolist()
                    prediction_list += prediction.detach().cpu().numpy().tolist()
                    patients_list+=patient_id.detach().cpu().numpy().tolist()
                    # for ind in range(len(patient_id)):
                    #     feature_dict[int(patient_id[ind])] = feature[ind, :].squeeze().detach().cpu().numpy().tolist()
            # with open(os.path.join(output_path, 'Patch_feature.json'), 'w') as f:
            #         json.dump(feature_dict, f)
            acc = accuracy_score(label_list, prediction_list)
            precision = precision_score(label_list, prediction_list)
            recall = recall_score(label_list, prediction_list)
            f1 = f1_score(label_list, prediction_list)

            fpr, tpr, threshold = roc_curve(label_list, score_list)
            roc = auc(fpr, tpr)
            auroc_pr(x=fpr, y=tpr, area_under_curve=roc, mode='roc', save_path=os.path.join(output_path, 'Figure'))

            precision_pr, recall_pr, threshold = precision_recall_curve(label_list, score_list)
            ap = average_precision_score(label_list, score_list)
            auroc_pr(x=recall_pr, y=precision_pr, area_under_curve=ap, mode='pr',
                     save_path=os.path.join(output_path, 'Figure'))

            print(f'{fold} accuracy: ', acc)
            print(f'{fold} precision: ', precision)
            print(f'{fold} recall: ', recall)
            print(f'{fold} f1: ', f1)
            p_roc,p_pr,p_acc,p_precision,p_recall,p_f1 = self.get_patient_results(patients_list,score_list,label_list,output_path,mode='cross')

            cross_result.loc[str(fold),'fold'] = str(fold)
            cross_result.loc[str(fold),'auroc'] = roc
            cross_result.loc[str(fold),'aupr'] = ap
            cross_result.loc[str(fold),'accuracy'] = acc
            cross_result.loc[str(fold), 'precision'] = precision
            cross_result.loc[str(fold), 'recall'] = recall
            cross_result.loc[str(fold), 'f1'] = f1

            cross_result.loc[str(fold),'p_auroc'] = p_roc
            cross_result.loc[str(fold),'p_aupr'] = p_pr
            cross_result.loc[str(fold),'p_accuracy'] = p_acc
            cross_result.loc[str(fold), 'p_precision'] = p_precision
            cross_result.loc[str(fold), 'p_recall'] = p_recall
            cross_result.loc[str(fold), 'p_f1'] = p_f1
        cross_result.to_csv(os.path.join(self.output_path,'CrossVal_result.csv'))
    def train(self):
        '''
        这个函数不是做交叉验证的，是做训练集和验证集的合并
        :return:
        '''
        #将训练集中的sorafenib患者删除
        train_index = self.split_data[str(0)][0]
        val_index = self.split_data[str(0)][1]
        train_index_noSorafenib = [i for i in train_index if len(i)>7]
        val_index_noSorafenib = [i for i in val_index if len(i)>7]

        train_loader = self.get_data(train_index_noSorafenib+val_index_noSorafenib,'train')

        net = self.get_model() #模型已经上了gpu
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(),lr = self.lr,betas=(0.9, 0.99))
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_loss = []
        train_acc = []
        loss_best = 100
        stop_index = 0
        for epoch in range(self.epochs):
            net,loss,accuracy = self.train_epoch(net,train_loader,criterion,optimizer)
            print(f'{epoch} {loss}')
            exp_lr_scheduler.step()
            train_loss.append(loss)
            train_acc.append(accuracy)
            if loss < loss_best:
                loss_best = loss
                best_model_wts = copy.deepcopy(net.state_dict())
                stop_index = 0
            else:
                stop_index += 1
            # if stop_index >= 5:
            #     break
        draw_LossAcc(epoch, 'train', train_loss, train_acc, self.output_path)
        model_save_path = os.path.join(self.output_path, 'Model')
        self.makefile(model_save_path)
        torch.save(best_model_wts, os.path.join(model_save_path, 'best.pkl'))
        return best_model_wts

    def train_epoch(self,net,train_loader,criterion,optimizer):
        label_list = []
        prediction_list = []
        score_list = []
        loss = 0
        for idx,sample in enumerate(train_loader):
            patho_fea,clinical_fea,therapy_fea = sample['patho_feature'],sample['clinical_feature'],sample['therapy_feature']
            label,patient_id = sample['label'],sample['patient_id']
            patho_fea = patho_fea.to(self.device)
            clinical_fea = clinical_fea.to(self.device)
            # clinical_contrast = torch.zeros_like(clinical_fea)
            therapy_fea = therapy_fea.to(self.device)
            output = net(patho_fea) 
            score = output[:,1] #score在训练过程是没用的
            prediction = torch.argmax(output,dim=-1)#用于求acc，画acc-loss曲线
            loss_iter = criterion(output, label.to(self.device))
            optimizer.zero_grad()
            loss_iter.backward()
            optimizer.step()
            loss += float(loss_iter)
            label_list += label.detach().cpu().numpy().tolist()
            prediction_list += prediction.detach().cpu().numpy().tolist()
            score_list += score.detach().cpu().numpy().tolist()
        loss = loss / (idx+1)
        accuracy = accuracy_score(label_list,prediction_list)
        return net,loss,accuracy

    def test(self,best_wts,test_csv):
        net = self.get_model() #模型已经上了gpu
        net.load_state_dict(best_wts)
        net.eval()
        test_file = pd.read_csv(test_csv)
        data_index_test = [i for j in test_file['ID'] for i in j.split(',') if len(i)>7]
        # data_index_test = [i for j in test_file['ID'] for i in j.split(',')]
        test_loader = self.get_data(data_index_test,'test')

        label_list = []
        score_list = []
        prediction_list = []
        patient_list = []
        feature_dict = {}
        wsi_score_dict = {}
        roc_dict = {}
        with torch.no_grad():
            for step,sample in enumerate(test_loader):
                patho_fea, clinical_fea, therapy_fea = sample['patho_feature'], sample['clinical_feature'], sample['therapy_feature']
                label, patient_id = sample['label'], sample['patient_id']
                wsi = sample['wsi']
                patho_fea = patho_fea.to(self.device)
                clinical_fea = clinical_fea.to(self.device)
                therapy_fea = therapy_fea.to(self.device)
                output = net(patho_fea)  #这是0416 直接用pathological feature来预测复发

                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                score = output[:, 1]
                prediction = torch.argmax(output, dim=-1)

                score_list += score.detach().cpu().numpy().tolist()
                label_list += label.detach().cpu().numpy().tolist()
                prediction_list += prediction.detach().cpu().numpy().tolist()
                patient_list += patient_id.detach().cpu().numpy().tolist()
        roc_dict['label_list'] = label_list
        roc_dict['score_list'] = score_list
        roc_dict['patients_list'] = patient_list

        acc = accuracy_score(label_list, prediction_list)
        precision = precision_score(label_list, prediction_list)
        recall = recall_score(label_list, prediction_list)
        f1 = f1_score(label_list, prediction_list)

        fpr, tpr, threshold = roc_curve(label_list, score_list)
        roc = auc(fpr, tpr)
        auroc_pr(x=fpr, y=tpr, area_under_curve=roc, mode='roc', save_path=os.path.join(self.output_path, 'Figure'))

        precision_pr, recall_pr, threshold = precision_recall_curve(label_list, score_list)
        ap = average_precision_score(label_list, score_list)
        auroc_pr(x=recall_pr, y=precision_pr, area_under_curve=ap, mode='pr',
                 save_path=os.path.join(self.output_path, 'Figure'))
        self.get_patient_results(patient_list,score_list,label_list,self.output_path)
        # # 画patch的混淆矩阵图
        # patch_confusion_matrix_path = os.path.join(self.output_path, 'confusion_matrix')
        # if not os.path.exists(patch_confusion_matrix_path):
        #     os.makedirs(patch_confusion_matrix_path)
        # draw_confusion_matrix(label_list, prediction_list, ['benign', 'malignant'], patch_confusion_matrix_path)
        print('WSI Result...')
        print('auroc: ',roc)
        print('aupr: ',ap)
        print('accuracy: ',acc)
        print('precision: ',precision)
        print('recall: ',recall)
        print('f1: ',f1)

    def get_patient_results(self,patients_list,score_list,label_list,output_path,mode='test'):
        patients_result = {}
        labels_list = []
        for i in range(len(patients_list)):
            if patients_list[i] not in list(patients_result.keys()):
                patients_result[patients_list[i]] = [score_list[i]]
                labels_list.append(label_list[i])
            else:
                patients_result[patients_list[i]].append(score_list[i])
        for key in patients_result.keys():
            patients_result[key] = sum(patients_result[key]) / len(patients_result[key])
        patients_score_list = [patients_result[key] for key in patients_result.keys()]


        patients_prediction_list = [1 if patients_score_list[i] >0.5 else 0 for i in range(len(patients_score_list))]
        fpr, tpr, threshold = roc_curve(labels_list, patients_score_list)
        roc = auc(fpr, tpr)
        auroc_pr(x=fpr, y=tpr, area_under_curve=roc, mode='roc', save_path=os.path.join(output_path, 'Patients_Figure'))

        precision_pr, recall_pr, threshold = precision_recall_curve(labels_list, patients_score_list)
        ap = average_precision_score(labels_list, patients_score_list)
        auroc_pr(x=recall_pr, y=precision_pr, area_under_curve=ap, mode='pr',
                 save_path=os.path.join(output_path, 'Patients_Figure'))


        acc = accuracy_score(labels_list, patients_prediction_list)
        precision = precision_score(labels_list, patients_prediction_list)
        recall = recall_score(labels_list, patients_prediction_list)
        f1 = f1_score(labels_list, patients_prediction_list)
        print('patients results: ')
        print('auroc: ',roc)
        print('aupr: ',ap)
        print('accuracy: ',acc)
        print('precision: ',precision)
        print('recall: ',recall)
        print('f1: ',f1)
        if mode == 'test':
            draw_confusion_matrix(labels_list,patients_prediction_list,['recurrence','non-recurrence'],self.output_path)
        elif mode == 'cross':
            return roc,ap,acc,precision,recall,f1


    def tcga_test(self,best_wts,batch_size):
        net = self.get_model() #模型已经上了gpu
        net.load_state_dict(best_wts)
        net.eval()


        label_path = './data/MultiFactor_prediction/two_recurrence_TCGA.xlsx'
        label_file = pd.read_excel(label_path)
        tcga_data_index = list(label_file['bcr_patient_barcode'])
        # tcga_data_index = os.listdir('./TCGA/02-final-TEST-Patch-ColorNormalization/')
        patho_path = './pathological_feature/02-TCGA_wsi_feature.json'
        # patho_path = './Results/13-resnet18-selfattn-softmax/0.5Scale-Results/stage2_TCGA_wsi_feature.json'
        dataset = TCGA_MultiFactor_Dataset(patho_path,label_path,tcga_data_index)
        data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)

        label_list = []
        score_list = []
        prediction_list = []
        tcga_survival_dict = {}
        roc_dict = {}
        i = 0
        with torch.no_grad():
            for step,sample in enumerate(data_loader):
                patho_fea, clinical_fea, therapy_fea = sample['patho_feature'], sample['clinical_feature'], sample['therapy_feature']
                label, patient_id = sample['label'], sample['patient_id']
                wsi = sample['wsi']
                patho_fea = patho_fea.to(self.device)
                clinical_fea = clinical_fea.to(self.device)
                # clinical_contrast = torch.zeros_like(clinical_fea)
                therapy_fea = therapy_fea.to(self.device)
                output = net(patho_fea)
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                score = output[:, 1]
                # for ind in range(len(wsi)):
                #     tcga_survival_dict[wsi[ind]] = float(score[ind])
                #从这里开始保存模型对tcga wsi预测分数的csv文件============
        #         for ind in range(len(wsi)):
        #             single_wsi = wsi[ind]
        #             _index = label_file.index[label_file.bcr_patient_barcode==single_wsi].tolist()[0]
        #             label_file.loc[_index,'final_recurrence_risk'] = score[ind].detach().cpu().numpy().tolist()
        # label_file.to_csv(os.path.join(self.output_path,'tcga_survival_df.csv'),index=False)

                prediction = torch.argmax(output, dim=-1)

                score_list += score.detach().cpu().numpy().tolist()
                label_list += label.detach().cpu().numpy().tolist()
                prediction_list += prediction.detach().cpu().numpy().tolist()
                roc_dict['label_list'] = label_list
                roc_dict['score_list'] = score_list

        acc = accuracy_score(label_list, prediction_list)
        precision = precision_score(label_list, prediction_list)
        recall = recall_score(label_list, prediction_list)
        f1 = f1_score(label_list, prediction_list)
        #
        # print(label_list,score_list)
        fpr, tpr, threshold = roc_curve(label_list, score_list)
        roc = auc(fpr, tpr)
        auroc_pr(x=fpr, y=tpr, area_under_curve=roc, mode='roc', save_path=os.path.join(self.output_path, 'TCGA_Figure'))

        precision_pr, recall_pr, threshold = precision_recall_curve(label_list, score_list)
        ap = average_precision_score(label_list, score_list)
        auroc_pr(x=recall_pr, y=precision_pr, area_under_curve=ap, mode='pr',
                 save_path=os.path.join(self.output_path, 'TCGA_Figure'))
        print('auroc: ',roc)
        print('aupr: ',ap)
        print('accuracy: ',acc)
        print('precision: ',precision)
        print('recall: ',recall)
        print('f1: ',f1)

    def generate_clinical_decision(self,wts,tcga = False):
        if not tcga:
            clinical_decision = pd.DataFrame()
            patch_path = './ALL_ColorNormalization_patch/'
            all_data_index = os.listdir(patch_path)

            model = self.get_model()
            model.load_state_dict(wts)
            model.eval()
            i = 0
            dataloader = self.get_data(all_data_index,'test',tcga=False,reverse=False)

            with torch.no_grad():
                for step, sample in enumerate(dataloader):
                    patho_fea, clinical_fea, therapy_fea = sample['patho_feature'], sample['clinical_feature'], \
                    sample['therapy_feature']
                    label, patient_id = sample['label'], sample['patient_id']
                    wsi = sample['wsi']
                    patho_fea = patho_fea.to(self.device)
                    clinical_fea = clinical_fea.to(self.device)
                    therapy_fea = therapy_fea.to(self.device)
                    output = model(patho_fea, clinical_fea, therapy_fea.unsqueeze(-1))
                    if len(output.shape) == 1:
                        output = output.unsqueeze(0)
                    score = output[:, 1]
                    for ind in range(len(wsi)):
                        clinical_decision.loc[i,'wsi'] = wsi[ind]
                        if len(wsi[ind]) <5:
                            clinical_decision.loc[i,'therapy'] = 'sorafenib'
                            clinical_decision.loc[i,'sorafenib_risk'] = score[ind].detach().cpu().numpy().tolist()
                        else:
                            clinical_decision.loc[i,'therapy'] = 'non'
                            clinical_decision.loc[i,'non_risk'] = score[ind].detach().cpu().numpy().tolist()
                        clinical_decision.loc[i,'label'] = label[ind].detach().cpu().numpy().tolist()
                        i+=1
            dataloader = self.get_data(all_data_index, 'test', tcga=False, reverse=True)
            with torch.no_grad():
                for step, sample in enumerate(dataloader):
                    patho_fea, clinical_fea, therapy_fea = sample['patho_feature'], sample['clinical_feature'], \
                    sample['therapy_feature']
                    label, patient_id = sample['label'], sample['patient_id']
                    wsi = sample['wsi']
                    patho_fea = patho_fea.to(self.device)
                    clinical_fea = clinical_fea.to(self.device)
                    therapy_fea = therapy_fea.to(self.device)
                    output = model(patho_fea, clinical_fea, therapy_fea.unsqueeze(-1))
                    if len(output.shape) == 1:
                        output = output.unsqueeze(0)
                    score = output[:, 1]
                    for ind in range(len(wsi)):
                        wsi_name = wsi[ind]
                        _index = clinical_decision.index[clinical_decision.wsi==wsi_name].tolist()[0]
                        clinical_decision.loc[_index,'reverse_score'] = score[ind].detach().cpu().numpy().tolist()

            clinical_decision.to_csv(os.path.join(self.output_path,'internal_clinical_decision.csv'),index=False)
        else:
            clinical_decision = pd.DataFrame()
            tcga_data_index = os.listdir(
                './TCGA/02-final-TEST-Patch-ColorNormalization/')
            label_path = './tcga_clinical_feature.xlsx'
            label_file = pd.read_excel(label_path)
            patho_path = './0.5Scale-Results/stage2_TCGA_wsi_feature.json'
            dataset = TCGA_MultiFactor_Dataset(patho_path, label_path, tcga_data_index,reverse=False)
            dataloader = DataLoader(dataset, batch_size=self.test_bs, shuffle=False)


            model = self.get_model()
            model.load_state_dict(wts)
            model.eval()
            i = 0

            with torch.no_grad():
                for step, sample in enumerate(dataloader):
                    patho_fea, clinical_fea, therapy_fea = sample['patho_feature'], sample['clinical_feature'], \
                        sample['therapy_feature']
                    label, patient_id = sample['label'], sample['patient_id']
                    wsi = sample['wsi']
                    patho_fea = patho_fea.to(self.device)
                    clinical_fea = clinical_fea.to(self.device)
                    therapy_fea = therapy_fea.to(self.device)
                    output = model(patho_fea, clinical_fea, therapy_fea.unsqueeze(-1))
                    if len(output.shape) == 1:
                        output = output.unsqueeze(0)
                    score = output[:, 1]
                    for ind in range(len(wsi)):
                        clinical_decision.loc[i, 'wsi'] = wsi[ind]
                        _index = label_file.index[label_file.bcr_patient_barcode==wsi[ind]].tolist()[0]
                        if label_file.iloc[_index,1] == 'sorafenib':
                            clinical_decision.loc[i, 'therapy'] = 'sorafenib'
                            clinical_decision.loc[i, 'sorafenib_risk'] = score[ind].detach().cpu().numpy().tolist()
                        else:
                            clinical_decision.loc[i, 'therapy'] = 'non'
                            clinical_decision.loc[i, 'non_risk'] = score[ind].detach().cpu().numpy().tolist()
                        clinical_decision.loc[i, 'label'] = label[ind].detach().cpu().numpy().tolist()
                        i+=1
            dataset = TCGA_MultiFactor_Dataset(patho_path, label_path, tcga_data_index, reverse=True)
            dataloader = DataLoader(dataset, batch_size=self.test_bs, shuffle=False)
            with torch.no_grad():
                for step, sample in enumerate(dataloader):
                    patho_fea, clinical_fea, therapy_fea = sample['patho_feature'], sample['clinical_feature'], \
                        sample['therapy_feature']
                    label, patient_id = sample['label'], sample['patient_id']
                    wsi = sample['wsi']
                    patho_fea = patho_fea.to(self.device)
                    clinical_fea = clinical_fea.to(self.device)
                    therapy_fea = therapy_fea.to(self.device)
                    output = model(patho_fea, clinical_fea, therapy_fea.unsqueeze(-1))
                    if len(output.shape) == 1:
                        output = output.unsqueeze(0)
                    score = output[:, 1]
                    for ind in range(len(wsi)):
                        wsi_name = wsi[ind]
                        _index = clinical_decision.index[clinical_decision.wsi == wsi_name].tolist()[0]
                        clinical_decision.loc[_index, 'reverse_score'] = score[ind].detach().cpu().numpy().tolist()

            clinical_decision.to_csv(os.path.join(self.output_path, 'externalTCGA_clinical_decision.csv'), index=False)


    def makefile(self,file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    seed_torch()
    tile_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/ALL_ColorNormalization_patch/'
    save_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/Results/15-Multi-WSI-Result/'
    # model_wts_path = '/home/liyixin/HCC/Results/06-multifactorial-prediction/02-0.25-resnet18-cross5-V2-pathoModel/Model/best.pkl'
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # save_patch_pathological_feature(tile_path,save_path,model_wts_path,device)

 #---------------------------------------------------

    integrate_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/integrate_dataset_clinical_feature.xlsx'
    # wsi_patho_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/pathological_feature/01-New_VadahaneColorNormalization_selfattn-softmax-Final_WSI_pathofeature.json'
    wsi_patho_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/pathological_feature/03-Contrast-Final_WSI_pathofeature.json'
    # dict_test = ['0.25','0.33','0.5','0.66','0.75','all']
    dict_test = ['all']
    for idx in dict_test:
        patient_cross_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/cross_json_new/split_train_val/crossVal.json'
        output_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/Results/19-final-recurrence/compare-end-to-end/'
        # print(patient_cross_path,output_path)
        MultiFactor_Classifier = MultiFactor_main_WSI_level(integrate_path,wsi_patho_path,patient_cross_path,128,128,0.001,50,output_path)
        best_wts = MultiFactor_Classifier.train()
        test_csv = '/2data/liyixin/HCC/data/MultiFactor_prediction/cross_json_new/split_test/split_test.csv'
        # best_wts = torch.load('/2data/liyixin/HCC/data/MultiFactor_prediction/Results/18-two-year-recurrence/Model/best.pkl')
        MultiFactor_Classifier.test(best_wts,test_csv)
        # MultiFactor_Classifier.generate_clinical_decision(best_wts,tcga=False)
        # MultiFactor_Classifier.tcga_test(best_wts,256)
        # MultiFactor_Classifier.cross_train()

