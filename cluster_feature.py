import json
import numpy as np
import pandas as pd
import random
import torch
import os
from model import Resnet18
from dataset import My_Dataset
from torch.utils.data import DataLoader

def get_testset_patch_featue(model_wts_path,cross_path,patch_path,output_path):
    '''
    该函数的作用是生成不同比例模型在各自测试集上的patch特征，用作后续聚类
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Resnet18(2)
    net.load_state_dict(torch.load(model_wts_path))
    net.to(device)

    with open(cross_path,'r') as f:
        cross_data = json.load(f)
    test_index = cross_data[str(0)][2]
    test_dataset = My_Dataset(patch_path,test_index,'test')
    test_loader = DataLoader(test_dataset,batch_size=512,shuffle=False)

    feature_dict = {}
    net.eval()
    with torch.no_grad():
        for sample in test_loader:
            image = sample['image']
            patch_name = sample['patch_name']
            feature,_ = net(image.to(device))
            for ind in range(len(patch_name)):
                feature_dict[patch_name[ind]] = feature[ind, :].squeeze().detach().cpu().numpy().tolist()

    with open(os.path.join(output_path,'TESTSet_Patch_feature.json'),'w') as f:
        json.dump(feature_dict,f)


class Get_cluster_feature():
    def __init__(self, feature_path, integrate_path):
        self.label = pd.read_excel(integrate_path)
        with open(feature_path, 'r') as f:
            self.feature = json.load(f)

    def get_full_feature(self):
        label = []
        # data = pd.DataFrame()
        data = np.empty((len(self.feature), 513), dtype=object)
        for i in range(len(self.feature)):
            patch_name = list(self.feature.keys())[i]
            slide_name = '_'.join(patch_name.split('_')[:-1])
            data[i, range(512)] = self.feature[patch_name]
            data[i, 512] = slide_name
            index = \
            [self.label.index[self.label.ID == i].tolist()[0] for i in self.label['ID'] if slide_name in i.split(',')][
                0]
            label.append(self.label.iloc[index, 3])
        x = np.array(data[:, :-1])
        label = np.array(label)
        slide_name_list = list(data[:, -1])
        return x, label, slide_name_list

    def get_full_wsi_feature(self):
        x, label, slide_name = self.get_full_feature()
        wsi_list = set(slide_name)
        wsi_feature = np.empty((len(wsi_list), 512), dtype=object)
        wsi_labels = []
        i = 0
        for wsi in wsi_list:
            _index = np.where(np.array(slide_name) == wsi)[0]
            mean_feature = np.mean(x[_index], axis=0)
            wsi_feature[i, range(512)] = mean_feature
            wsi_labels.append(label[_index[0]])
            i += 1
        wsi_feature = np.array(wsi_feature)
        wsi_labels = np.array(wsi_labels)

        return wsi_feature, wsi_labels

    def get_random_wsi_list(self):
        wsi_list = ['_'.join(i.split('_')[:-1]) for i in list(self.feature.keys())]
        wsi_list = list(set(wsi_list))
        wsi_random_list = np.random.choice(wsi_list, 49, replace=False)
        return wsi_random_list


    def get_random_patch_feature(self, wsi_random_list):
        x, labels, slide_list = self.get_full_feature()

        data = np.empty((50000, 512))
        label = []
        test_id = 0
        j = 0
        for i in range(len(self.feature)):
            wsi_name = slide_list[i]
            patch_name = list(self.feature.keys())[i]
            if wsi_name in wsi_random_list:
                test_id += 1
                data[j, range(512)] = self.feature[patch_name]
                index = [self.label.index[self.label.ID == i].tolist()[0] for i in self.label['ID'] if
                         wsi_name in i.split(',')][0]
                label.append(self.label.iloc[index, 3])
                j += 1
                # label.append(labels[i])
        x = np.array(data[:test_id])
        label = np.array(label[:test_id])
        return x, label

    def get_random_wsi_feature(self, wsi_random_list):
        x, labels, slide_list = self.get_full_feature()
        data = np.empty((49, 512))
        label = []
        for i in range(len(wsi_random_list)):
            _index = np.where(np.array(slide_list) == wsi_random_list[i])[0]
            mean_feature = np.mean(x[_index], axis=0)
            data[i, range(512)] = mean_feature
            label.append(labels[_index[0]])
        x = np.array(data)
        label = np.array(label)
        return x, label
if __name__ == '__main__':
    resnet18_result_path = './05-resnet18-diffratio-bs128-pathoModel/'
    cross_path = './cross_json_new/'
    patch_path = './ALL_ColorNormalization_patch/'
    scales = ['0.25','0.33','0.5','0.66','0.75','all']
    for scale in scales:
        model_wts_path = os.path.join(resnet18_result_path,scale+'Scale-Results','Model','best.pkl')
        cross_path_scale =  os.path.join(cross_path,scale+'crossVal.json')
        output_path = os.path.join(resnet18_result_path,scale+'Scale-Results')
        get_testset_patch_featue(model_wts_path,cross_path_scale,patch_path,output_path)