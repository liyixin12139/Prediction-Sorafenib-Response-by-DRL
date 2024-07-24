import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_recall_curve,average_precision_score,auc,roc_curve,precision_score,recall_score,f1_score,roc_auc_score
import json
def draw_LossAcc(epoch,mode,loss_list,acc_list,save_path):
    if mode == 'train' and acc_list is not None:
        plt.figure()
        plt.cla()
        plt.title(f"Training Process",fontsize=20)


        plt.plot(range(epoch+1),loss_list,'r',label = 'train loss')
        plt.plot(range(epoch+1),acc_list,'b',label = 'train accuracy')
        plt.legend()
        output = os.path.join(save_path,'Train Figure')
        if not os.path.exists(output):
            os.makedirs(output)
        output_file = os.path.join(output,'training_process.png')
        plt.savefig(output_file)
        plt.close()
    elif mode == 'validation' and acc_list is not None:
        plt.figure()
        plt.cla()
        plt.title(f"Validate process",fontsize=20)
        plt.plot(range(epoch + 1), loss_list, 'r', label='validation loss')
        plt.plot(range(epoch + 1), acc_list, 'b', label='validation accuracy')
        plt.legend()
        output = os.path.join(save_path,'Validation Figure')
        if not os.path.exists(output):
            os.makedirs(output)
        output_file = os.path.join(output, 'validation_process.png')
        plt.savefig(output_file)
        plt.close()
    elif mode == 'train' and acc_list is None:
        plt.figure()
        plt.cla()
        plt.title(f'Training Process',fontsize = 20)
        plt.plot(range(epoch+1),loss_list,'r')
        # plt.legend()
        output = os.path.join(save_path,'Train Figure')
        if not os.path.exists(output):
            os.makedirs(output)
        output_file = os.path.join(output,'training_process.png')
        plt.savefig(output_file)
        plt.close()
def auroc_pr(x,y,area_under_curve,mode,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if mode=='roc':
        roc_auc = area_under_curve
        fpr = x
        tpr = y
        plt.figure()
        lw = 2
        plt.subplot(1,1,1)
        plt.plot(fpr,tpr,color='darkorange',
                 lw=lw,label='ROC curve (area=%0.3f)' % roc_auc)
        plt.rcParams.update({'font.size': 15})
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],color = 'navy',lw=lw,linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('1-specificity')
        plt.ylabel('sensitivity')
        plt.legend(loc='lower right')
        plt.title('Test ROC Curve')
        plt.savefig(os.path.join(save_path,'test_roc.png'))
        plt.close()
    elif mode =='pr':
        ap = area_under_curve
        recall = x
        precision = y
        plt.figure()
        lw = 2
        plt.subplot(1, 1, 1)
        plt.plot(recall, precision, color='darkorange',
                 lw=lw, label='PR curve (area = %0.3f)' % ap)
        plt.rcParams.update({'font.size': 15})
        plt.legend(loc='lower right')
        # plt.plot([0,1],[0,1],color = 'navy',lw=lw,linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall',fontsize=10)
        plt.ylabel('Precision',fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='lower right')
        plt.title('Test PR Curve')
        plt.savefig(os.path.join(save_path,'test_pr.png'))
        plt.close()

def draw_confusion_matrix(label,pred,label_name,output_path):
    conf_matrix = confusion_matrix(pred,label)
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2	#数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(2):
        for y in range(2):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     fontsize = 20,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()#保证图不重叠
    plt.yticks(range(2), label_name,rotation=90)
    plt.xticks(range(2), label_name)#X轴字体倾斜45°
    plt.savefig(os.path.join(output_path,'test_confusion_matrix.png'))
    plt.close()

def multi_roc(y_tests,y_preds,names,output_path):
    plt.figure(0).clf()
    for i in range(len(names)):
        y_test = y_tests[i]
        y_pred = y_preds[i]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
        plt.plot(fpr,tpr,label=f"{names[i]}, AUC="+str(auc))
    plt.plot([0,1],[0,1],color = 'navy',lw=2,linestyle='--')
    plt.xlabel('1-specificity')
    plt.ylabel('sensitivity')
    plt.legend()
    plt.savefig(os.path.join(output_path,'TCGA_MultiROC.png'))
    plt.close()

def multi_pr(y_tests,y_preds,names,output_path):
    from sklearn.metrics import precision_recall_curve,average_precision_score
    plt.figure(0).clf()
    for i in range(len(names)):
        y_test = y_tests[i]
        y_pred = y_preds[i]
        precision_pr, recall_pr, threshold = precision_recall_curve(y_test, y_pred)
        ap = round(average_precision_score(y_test, y_pred),4)
        plt.plot(recall_pr,precision_pr,label=f"{names[i]}, AUC="+str(ap))
    plt.ylim((0,1))
    plt.xlabel('1-specificity')
    plt.ylabel('sensitivity')
    plt.legend()
    plt.savefig(os.path.join(output_path,'TCGA_MultiPR.png'))
    plt.close()


