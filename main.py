import os
import sys
sys.path.append('/home/jchen152/workspace/Deep_Insigth_Project/Zero_Shot/S3D_HowTo100M')
from s3dg import S3D
import numpy as np
import torch
from DataLoader.Dataloader_newcls import get_sets
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import random
import argparse
import sys

from util.get_acc import cal_cfm


# ============== Get Configuration =================
def get_arg():
    cfg=argparse.ArgumentParser()
    cfg.add_argument('--exp_name',default='fold0_optimize_text_of_normaldriving')
    cfg.add_argument('--epochs',default=100)
    cfg.add_argument('--train',action='store_true',default=True)
    
    cfg.add_argument('--data_path',default='/home/jchen152/workspace/Deep_Insigth_Project/Zero_Shot/Data/synDD_1fps')
    cfg.add_argument('--project_path',default='/home/jchen152/workspace/Deep_Insigth_Project/Zero_Shot')

    cfg.add_argument('--batch_size',default=5,type=int)
    cfg.add_argument('--fold',default=0,type=int)
    cfg.add_argument('--video_based',default=True)
   
    
    return cfg.parse_args()
cfg=get_arg()




# ======= create txt ==========
# label_map = {
# 	0:	"Normal Driving",   # *
# 	1:	"Drinking",                 # *
# 	2:	"Phone Call right",        # *
# 	3:	"Phone Call left",         # *
# 	4:	"Eating",
# 	5:	"Text Left",
# 	6:	"Text reft",
# 	7:	"Hair or makeup",
# 	8:	"Reaching behind",
# 	9:	"Adjust control panel",
# 	10:	"Pick up from floor (Driver)",
# 	11:	"Pick up from floor (Passenger)",
# 	12:	"Talk to passenger at the right",
# 	13:	"Talk to passenger at backseat",
# 	14:	"yawning",
# 	15:	"Hand on head",
# 	16:	"Singing with music",
# 	17:	"shaking or dancing with music",
# }
# text_list=list(label_map.values())[:8]



class_mapping = {
    0:"driver is driving with hands on the steering wheel",
    1:"driver is adjusting his or her hair while driving a car",
    2:"driver is drinking water from a bottle while driving a car",
    3:"driver is eating while driving a car",
    4:"driver is picking something from floor while driving a car",
    5:"driver is reaching behind to the backseat while driving a car",
    6:"driver is singing a song with music and smiling wiled driving",
    7:"driver is talking to the phone on hand while driving a car",
    8:"driver is yawning while driving a car"}


# class_mapping = {
#     0:"driver is driving with hands on the steering wheel",
#     1:"driver is adjusting his or her hair while driving a car",
#     2:"driver is drinking water from a bottle while driving a car",
#     3:"driver is eating while driving a car",
#     4:"driver is picking something from floor while driving a car",
#     5:"driver is reaching behind to the backseat while driving a car",
#     6:"driver is talking to the phone on hand or singing while driving a car",
#     7:"driver is yawning while driving a car"}


# class_mapping = {
#     0:"driver is adjusting his or her hair while driving a car",
#     1:"driver is drinking water from a bottle while driving a car",
#     2:"driver is eating while driving a car",
#     3:"driver is picking something from floor while driving a car",
#     4:"driver is reaching behind to the backseat while driving a car",
#     5:"driver is singing a song with music and smiling wiled driving",
#     6:"driver is talking to the phone on hand while driving a car"}
text_list=list(class_mapping.values())

# =================================



def main():
    os.environ['CUDA_LAUNCH_BLOCKING']='1'

    seed=0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled=False
  
    cuda=0
    model=S3D('/home/jchen152/workspace/Deep_Insigth_Project/Zero_Shot/S3D_HowTo100M/s3d_dict.npy', 512)
    model.load_state_dict(torch.load('/home/jchen152/workspace/Deep_Insigth_Project/Zero_Shot/S3D_HowTo100M/s3d_howto100m.pth'))
    train_loader,valid_loader=get_sets(data_path=cfg.data_path,batch_size=cfg.batch_size,fold=cfg.fold,video_based=cfg.video_based)
    
    if cfg.train:
        train_model(model,train_loader,valid_loader,cfg.exp_name,cuda)
    
    else:
        # pth_folder=os.path.join('./pth_file',exp_name)
        # test(model,test_loader,pth_folder=pth_folder)
        print('not implemented yet')
        
def test(model,data_loader,pth_folder):
    pass



def train_model(model,train_loader,valid_loader,exp_name,cuda_n):
    assert torch.cuda.is_available()
    epoch_acc=[]

    #这里应该用GPU
    device=torch.device('cuda:{}'.format(cuda_n))
    model=model.to(device)


    initial_epoch=0
    training_epoch=cfg.epochs

    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=3*0.0001)
    lr_schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.arange(10,training_epoch,30),gamma=0.7)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001*100, momentum=0.9, weight_decay=1e-4)
    # lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training_epoch, eta_min=0.001)


    
    # lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training_epoch, eta_min=0.001)



    #here we define train_one_epoch
    def train_one_epoch():
        iterations=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        
        #真正训练这里应该解封
        epsum=run_one_epoch(model,iterations,"train",loss_func=loss_func,optimizer=optimizer,loss_interval=10)
        
        summary={"loss/train":np.mean(epsum['losses'])}
        return summary


    def eval_one_epoch():
        iteration=tqdm(valid_loader,ncols=100,unit='batch',leave=False)
        #epsum only have logit and labes
        #epsum['logti'] is (batch,4096,13)
        #epsum['labels] is (batch,4096)
        
        epsum=run_one_epoch(model,iteration,"valid",loss_func=loss_func)
        mean_acc=np.mean(epsum['acc'])
        epoch_acc.append(mean_acc)
        
        summary={'meac':mean_acc}
        summary["loss/valid"]=np.mean(epsum['losses'])
        return summary,epsum['cfm']



    #build tensorboard
    
    # tensorboard=SummaryWriter(log_dir='.Exp/{}/TB'.format(exp_name))
    tqdm_epoch=tqdm(range(initial_epoch,training_epoch),unit='epoch',ncols=100)

    #build folder for pth_file
    exp_path=os.path.join(cfg.project_path,'./Exp',exp_name)
    pth_path=os.path.join(exp_path,'pth_file')
    tensorboard_path=os.path.join(exp_path,'TB')
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    if not os.path.exists(pth_path):
        os.mkdir(pth_path)
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)
    # pth_save_path=os.path.join('Exp',exp_name,'pth_file')
    # if not os.path.exists(pth_save_path):
    #     os.mkdir(pth_save_path)
    
    tensorboard=SummaryWriter(log_dir=tensorboard_path)


    for e in tqdm_epoch:
        train_summary=train_one_epoch()
        valid_summary,confusion_mat=eval_one_epoch()
        summary={**train_summary,**valid_summary}
        lr_schedule.step()
        #save checkpoint
        if np.max(epoch_acc)==epoch_acc[-1]:
            summary_saved={**summary,
                            'model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict(),
                            'cm':confusion_mat,
                            'cls_map':class_mapping}


            # torch.save(summary_saved,'./pth_file/{0}/epoch_{1}'.format(exp_name,e))
            torch.save(summary_saved,os.path.join(pth_path,'epoch_{}'.format(e)))
        
        for name,val in summary.items():
            tensorboard.add_scalar(name,val,e)
    


def run_one_epoch(model,tqdm_iter,mode,loss_func=None,optimizer=None,loss_interval=10):
    if mode=='train':
        model.train()
    else:
        model.eval()
        param_grads=[]
        for param in model.parameters():
            param_grads+=[param.requires_grad]
            param.requires_grad=False
            
    confusion_mat=np.zeros((len(text_list),len(text_list)))
    summary={"losses":[],"acc":[]}
    device=next(model.parameters()).device

    for i,(x_cpu,y_cpu) in enumerate(tqdm_iter):
        x,y=x_cpu.to(device),y_cpu.to(device)

        if mode=='train':
            optimizer.zero_grad()
            
        #logtis' shape is [batch,40]
        #y size is [batch,1]
        
        video_output=model(x)
        text_output=model.text_module(text_list)
        video_embedding = video_output['video_embedding']
        text_embedding = text_output['text_embedding']
        logits = torch.matmul(text_embedding, video_embedding.t()).t()


        if loss_func is not None:
            #### here is the loss #####
            loss=loss_func(logits,y)
            #loss=loss_func(re_logit,y.view(-1))
            summary['losses']+=[loss.item()]
        
        if mode=='train':
            loss.backward(retain_graph=False)
            optimizer.step()

            #display
            if loss_func is not None and i%loss_interval==0:
                tqdm_iter.set_description("Loss: {:.3f}".format(np.mean(summary['losses'])))

        else:
            # log=logits.cpu().detach().numpy()
            # lab=y_cpu.numpy()
            
            batch_cf=cal_cfm(logits,y_cpu,len(text_list))
            confusion_mat+=batch_cf
            batch_acc=np.trace(batch_cf)/np.sum(batch_cf)
            summary['acc'].append(batch_acc)

        
        # summary['logits']+=[logits.cpu().detach().numpy()]
        # summary['labels']+=[y_cpu.numpy()]
            if i%loss_interval==0:
                tqdm_iter.set_description("mea_ac: %.3f"%(np.mean(summary['acc'])))


    if mode!='train':
        for param,value in zip(model.parameters(),param_grads):
                param.requires_grad=value

        summary['cfm']=confusion_mat
        
    # summary["logits"] = np.concatenate(summary["logits"], axis=0)
    # summary["labels"] = np.concatenate(summary["labels"], axis=0)

    return summary


if __name__=='__main__':
    # parser=get_parse()
    # data_path='/data1/jiajing/dataset/plane_seg_sample'
    main()
