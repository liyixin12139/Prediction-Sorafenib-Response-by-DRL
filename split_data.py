import os
import json
import pandas as pd
import numpy as np
import random
import torch
def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def patho_generate_CrossVal_json(path,test_rate,val_fold,save_path,ratio):
    '''
    这个函数用于将筛选出来的良恶患者csv文件，混合到一起划分五折交叉验证json文件
    根据patient-id进行五折交叉，然后将对应的wsi文件保存
    :param test_rate:
    :param val_fold: 交叉验证折数，eg 5
    :param save_path: 保存json的路径
    :return:{'0':[[训练],[验证],[测试]]}
    '''
    file = pd.read_csv(path)
    cross_dict = {}
    patients_dict = {}
    for i in range(len(file)):
        patients_dict[file.iloc[i,0]] = [file.iloc[i,2],file.iloc[i,3]]


    test_patients = list(np.random.choice(list(file['patient_id']),round(len(file)*test_rate),replace=False))
    test_slides = [x for id in test_patients for x in patients_dict[id][0].split(',')]
    test_labels = [patients_dict[id][1] for id in test_patients]
    train_val_patients = list(set(list(file['patient_id'])) - set(test_patients))
    random.shuffle(train_val_patients)

    while True:
        i = 0
        val_labels_num = []
        for fold in range(val_fold):
            if fold != val_fold-1:
                val_patients = train_val_patients[i:i+(len(train_val_patients)//val_fold)]
            else:
                val_patients = train_val_patients[i:]
            train_patients = list(set(train_val_patients) - set(val_patients))

            i += len(train_val_patients) // val_fold
            val_labels = [patients_dict[id][1] for id in val_patients]
            val_labels_num.append(len(np.unique(val_labels)))
            train_slides = [x for id in train_patients for x in patients_dict[id][0].split(',')]
            val_slides = [x for id in val_patients for x in patients_dict[id][0].split(',')]
            cross_dict[fold] = [train_slides,val_slides,test_slides]
        if len(np.unique(val_labels_num)) ==1:
            with open(os.path.join(save_path,ratio+'crossVal.json'),'w') as f:
                json.dump(cross_dict,f)
            break


if __name__ == '__main__':
    seed_torch()
    # ratios = ['0.25','0.33','0.5','0.66','0.75','all']
    ratios = ['0.25', '0.5',  '0.75', 'all']
    select_path = './data/MultiFactor_prediction/子数据集筛选结果/'
    for i in range(len(ratios)):
        ratio = ratios[i]
        select_path_ratio = os.path.join(select_path,ratio+'.csv')
        save_path = './data/MultiFactor_prediction/cross_json/'
        patho_generate_CrossVal_json(select_path_ratio,0.1,5,save_path,ratio)

