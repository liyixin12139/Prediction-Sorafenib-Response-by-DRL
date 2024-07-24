#主要包括数据集的类以及生成和整理病理特征函数
import json
import numpy as np
import pandas as pd
import random
import torch
import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from model import CNN_SelfAttention_attn,CNN_SelfAttention_softmax




class My_Dataset(Dataset):
    def __init__(self,patch_path,data_index,mode):
        super(My_Dataset,self).__init__()
        self.label = pd.read_excel('/2data/liyixin/HCC/data/MultiFactor_prediction/integrate_dataset_clinical_feature.xlsx')
        self.mode = mode
        self.patch_list = []
        for single_slide in data_index:
            for singel_patch in os.listdir(os.path.join(patch_path,single_slide)):
                self.patch_list.append(os.path.join(patch_path,single_slide,singel_patch))

        self.train_process = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __getitem__(self, item):
        patch_name = os.path.basename(self.patch_list[item].split('.')[0])
        slide_name = '_'.join(os.path.basename(self.patch_list[item]).split('_')[:-1])
        _index = [self.label.index[self.label.ID == i].tolist()[0] for i in self.label.ID if slide_name in i.split(',')][0]
        label = torch.tensor(self.label.iloc[_index,3])   #这里的label代表的是病理图像代表的预后良恶性，其实是我们约等于的label
        image = Image.open(self.patch_list[item])
        if self.mode == 'train':
            image = self.train_process(image)
        else:
            image = self.test_process(image)
        return {'image':image,
                'label':label,
                'patch_name':patch_name,
                'slide_name':slide_name}
    def __len__(self):
        return len(self.patch_list)



class TCGA_patho_dataset(Dataset):
    def __init__(self,tcga_path,label_path):
        super(TCGA_patho_dataset,self).__init__()
        self.patch_list = []
        self.label = pd.read_excel(label_path)
        data_index = list(self.label['bcr_patient_barcode'])
        for slide in data_index:
            if slide in os.listdir(tcga_path):
                for patch in os.listdir(os.path.join(tcga_path,slide)):
                    self.patch_list.append(os.path.join(tcga_path,slide,patch))

        self.test_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __getitem__(self, item):
        slide_name = os.path.basename(os.path.split(self.patch_list[item])[-2])
        patch_name = slide_name+'_'+os.path.basename(self.patch_list[item]).split('.')[0]
        _index = self.label.index[self.label.bcr_patient_barcode == slide_name].tolist()[0]
        label = torch.tensor(self.label.iloc[_index,3])  #这里到时候要根据预测的年限来换
        patch = self.test_process(Image.open(self.patch_list[item]))
        return {
            'image':patch,
            'label':label,
            'slide_name':slide_name,
            'patch_name':patch_name
        }
    def __len__(self):
        return len(self.patch_list)



class WSI_MultiFactor_Dataset(Dataset):
    def __init__(self,patho_fea_path,integrate_path,data_index,reverse = False):
        super(WSI_MultiFactor_Dataset,self).__init__()
        self.data = pd.read_excel(integrate_path) #从这里得到临床特征
        self.wsi_list = data_index
        self.clinical_index = [10, 11, 12, 15, 16, 19, 20]
        # self.clinical_index = [15, 16,20]
        self.clincial_data = np.array(self.data.iloc[:,self.clinical_index])
        scaler = MinMaxScaler(feature_range=(0,1))
        self.clincial_data = scaler.fit_transform(self.clincial_data)
        with open(patho_fea_path,'r') as f: #从这里得到病理特征
            self.patho_feature = json.load(f)
        self.reverse = reverse
    def __getitem__(self, item):
        wsi = self.wsi_list[item]

        _index = [self.data.index[self.data.ID == j].tolist()[0] for j in self.data['ID'] for i in j.split(',') if wsi == i][0]
        patho_fea = torch.tensor(self.patho_feature[wsi]) #所以最终病理模型生成所有patch的病理特征后要做全局平均，然后以WSI为key保存到字典中
        clinical_fea = torch.tensor(self.clincial_data[_index,:])
        #将临床特征全变成0来做仅病理特征的输出
        # clinical_fea = torch.zeros(7)
        if not self.reverse:
            therapy_fea = torch.tensor(1. if self.data.iloc[_index,1] == 'sorafenib' else 0.)
        else:
            therapy_fea = torch.tensor(0. if self.data.iloc[_index, 1] == 'sorafenib' else 1.)
        label = self.data.iloc[_index,3] #5代表一年内的复发状况,6是2年，3代表最终复发
        patient_id = self.data.iloc[_index,0]
        return {
            'patho_feature':patho_fea,
            'clinical_feature':clinical_fea,
            'therapy_feature':therapy_fea,
            'label':label,
            'patient_id':patient_id,
            'wsi':wsi
        }
    def __len__(self):
        return len(self.wsi_list)


class TCGA_MultiFactor_Dataset(Dataset):
    def __init__(self,patho_fea_path,clinical_path,data_index,reverse = False):
        super(TCGA_MultiFactor_Dataset,self).__init__()
        self.data = pd.read_excel(clinical_path) #从这里得到临床特征
        self.wsi_list = data_index
        self.clinical_index = [11,12,13,14,15,16,17]
        self.clincial_data = np.array(self.data.iloc[:,self.clinical_index])
        scaler = MinMaxScaler(feature_range=(0,1))
        self.clincial_data = scaler.fit_transform(self.clincial_data)
        with open(patho_fea_path,'r') as f: #从这里得到病理特征
            self.patho_feature = json.load(f)
        self.reverse = reverse
    def __getitem__(self, item):
        wsi = self.wsi_list[item]

        _index = [self.data.index[self.data.bcr_patient_barcode == j].tolist()[0] for j in self.data['bcr_patient_barcode']  if wsi == j][0]
        patho_fea = torch.tensor(self.patho_feature[wsi]) #所以最终病理模型生成所有patch的病理特征后要做全局平均，然后以WSI为key保存到字典中
        clinical_fea = torch.tensor(self.clincial_data[_index,:])
        if not self.reverse:
            therapy_fea = torch.tensor(1. if self.data.iloc[_index,1] == 'sorafenib' else 0.)
        else:
            therapy_fea = torch.tensor(0. if self.data.iloc[_index, 1] == 'sorafenib' else 1.)
        label = self.data.iloc[_index,3] #代表一年内的复发状况
        patient_id = self.data.iloc[_index,0]
        return {
            'patho_feature':patho_fea,
            'clinical_feature':clinical_fea,
            'therapy_feature':therapy_fea,
            'label':label,
            'patient_id':patient_id,
            'wsi':wsi
        }
    def __len__(self):
        return len(self.wsi_list)




def save_pathofeature(model_name,model_wts_path,patch_path,save_path,batch_size):
    '''
    这个函数是当最终的病理模型以及对应的ratio选定之后，用于生成所有patch和wsi的病理特征，
    用于在第二步多模态模型构建中提供最终的病理特征
    Args:
        model_name:
        model_wts_path:
        save_path:
    Returns: patch_feature.json, wsi_feature.json
    '''
    #目前暂时用selfattn-attn的all模型试一下
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_index = [i for i in os.listdir(patch_path)]
    # print(data_index)
    dataset = My_Dataset(patch_path,data_index,'test')
    data_loader = DataLoader(dataset,batch_size = batch_size,shuffle=False)

    model_dict = {'cnn_selfattn_attn':CNN_SelfAttention_attn,
                  'cnn_selfattn_softmax':CNN_SelfAttention_softmax}
    model = model_dict[model_name]('resnet18',2)
    model.load_state_dict(torch.load(model_wts_path))
    model.to(device)
    model.eval()

    patch_feature_dict = {}
    with torch.no_grad():
        for sample in data_loader:
            image,patch_name,slide_name = sample['image'],sample['patch_name'],sample['slide_name']
            feature,_ = model(image.to(device))
            for i in range(len(patch_name)):
                patch_feature_dict[patch_name[i]] = feature[i,:].squeeze().detach().cpu().numpy().tolist()

    with open(os.path.join(save_path,'Final_Patch_PathoFea.json'),'w') as f:
        json.dump(patch_feature_dict,f)

    #接下来该保存图像水平的特征了
    wsi_dict = {}
    for ind in range(len(patch_feature_dict)):
        patch_name = list(patch_feature_dict.keys())[ind]
        wsi_name = '_'.join(patch_name.split('_')[:-1])
        if wsi_name not in wsi_dict.keys():
            wsi_dict[wsi_name] = [patch_feature_dict[patch_name]]
        else:
            wsi_dict[wsi_name].append(patch_feature_dict[patch_name])

    for key in wsi_dict.keys():
        mean_feature = np.mean(wsi_dict[key],axis=0)
        wsi_dict[key] = mean_feature.tolist()
    with open(os.path.join(save_path,'Final_WSI_pathofeature.json'),'w') as f:
        json.dump(wsi_dict,f)



def generate_stage2_tcgaPathofea(model_name,wts_path,out_num,tcga_path,tcga_label_path,save_path):
    '''
    用于将生成的TCGA的patch级特征全局平均后转为图像级病理特征
    Args:
        tcga_patho_fea_path:
    Returns: {'TCGA-xxxx-xxxx':[1,1,1,1,1...],
    }
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_dict = {'cnn_selfattn_softmax':CNN_SelfAttention_softmax}
    model = model_dict[model_name]('resnet18',out_num)
    model.load_state_dict(torch.load(wts_path))
    model.to(device)
    model.eval()

    dataset = TCGA_patho_dataset(tcga_path,tcga_label_path)
    data_loader = DataLoader(dataset,batch_size=128,shuffle=False)
    patch_feature_dict = {}

    with torch.no_grad():
        for sample in data_loader:
            image,patch_name = sample['image'],sample['patch_name']
            feature,_ = model(image.to(device))
            for ind in range(len(patch_name)):
                patch_feature_dict[patch_name[ind]] = feature[ind,:].squeeze().detach().cpu().numpy().tolist()
    with open(os.path.join(save_path,'stage2_TCGA_patch_feature.json'),'w') as f:
        json.dump(patch_feature_dict,f)


    wsi_feature_dict = {}
    for key in patch_feature_dict.keys():
        tcga_name = key.split('_')[0]
        if tcga_name not in wsi_feature_dict.keys():
            wsi_feature_dict[tcga_name] = [patch_feature_dict[key]]
        else:
            wsi_feature_dict[tcga_name].append(patch_feature_dict[key])
    for key in wsi_feature_dict.keys():
        mean_feature = np.mean(wsi_feature_dict[key],axis=0)
        wsi_feature_dict[key] = mean_feature.tolist()
    with open(os.path.join(save_path,'stage2_TCGA_wsi_feature.json'),'w') as f:
        json.dump(wsi_feature_dict,f)



if __name__ == '__main__':
    #从label-feature的csv文件出发，生成wsi的交叉验证json文件

    #将tcga的patch级特征转为wsi级特征
    # tcga_fea_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/Results/13-resnet18-selfattn-softmax/0.5Scale-Results/TCGA_Patch_feature.json'
    # save_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/Results/13-resnet18-selfattn-softmax/0.5Scale-Results/'
    # convert_patchfea2wsi(tcga_fea_path,save_path)


    #保存特定模型的病理特征
    # model_name = 'cnn_selfattn_softmax'
    # wts_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/Results/13-resnet18-selfattn-softmax/0.5Scale-Results/Model/best.pkl'
    # save_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/pathological_feature/'
    # batch_size = 256
    # save_pathofeature(model_name,wts_path,patch_path,save_path,batch_size)

    #生成第二阶段TCGA的patch和wsi病理特征
    # label_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/tcga_clinical_feature.xlsx'  #根据label文件来对其中的tcga生成特征
    # model_name = 'cnn_selfattn_softmax'
    # wts_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/Results/13-resnet18-selfattn-softmax/0.5Scale-Results/Model/best.pkl'
    # tcga_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/TCGA/0416-分开预测预后和索拉非尼-tcga数据/04-FINAL-all-negative-tcga-177/'
    # save_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/pathological_feature/'
    # generate_stage2_tcgaPathofea(model_name,wts_path,2,tcga_path,label_path,save_path)



    # label_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/tcga_clinical_feature.xlsx'
    # # cross_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/cross_json_new/0.25crossVal.json'
    # # tcga_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/TCGA/01-patho-EXTest-afterScreen/'
    # tcgalabel_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/tcga_clinical_feature.xlsx'
    #
    # tcga_fea_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/Results/13-resnet18-selfattn-softmax/0.5Scale-Results/TCGA_wsi_feature.json'
    # clinical_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/tcga_clinical_feature.xlsx'
    # tcga_file = pd.read_excel(clinical_path)
    # patho_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/Results/13-resnet18-selfattn-softmax/0.5Scale-Results/stage2_TCGA_wsi_feature.json'
    # data_index = os.listdir('/2data/liyixin/HCC/data/MultiFactor_prediction/TCGA/02-final-TEST-Patch-ColorNormalization/')
    # dataset = TCGA_MultiFactor_Dataset(patho_path,clinical_path,data_index)
    # for item in dataset:
    #     patho,clinical,therapy = item['patho_feature'],item['clinical_feature'],item['therapy_feature']
    #     if torch.isnan(clinical).any():
    #         print(item)

#生成patch的病理特征json文件============
    patch_path = './New_All_ColorNormalization_sorafenib_vahadane/'
    model_name = 'cnn_selfattn_softmax'
    wts_path = './Model/best.pkl'
    # wts_path = '/2data/liyixin/HCC/data/MultiFactor_prediction/Results/13-resnet18-selfattn-softmax/0.5Scale-Results/Model/best.pkl'
    save_path = './pathological_feature/'
    batch_size = 512
    save_pathofeature(model_name,wts_path,patch_path,save_path,batch_size)