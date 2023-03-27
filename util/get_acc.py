import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def cal_cfm(pred,label,ncls):
    pred=pred.cpu().detach().numpy()
    label=label.cpu().detach().numpy()
    
    pred=np.argmax(pred,-1)
    cfm=confusion_matrix(label,pred,labels=np.arange(ncls))
    return cfm



def get_topn_accuracy(pred,label,n):
    pred_n=torch.argsort(pred,-1,descending=True)[:,:n]
    right_pred_num=0
    for l,p in zip(pred_n,label):
        if p in l:
            right_pred_num+=1
            
    accuracy=right_pred_num/len(label)
    return accuracy




if __name__=='__main__':
    logtis=torch.randn((5,8))
    label=torch.arange(5)
    get_topn_accuracy(logtis,label,3)