import torch
import torch.nn as nn
import sys
sys.path.append('/home/jchen152/workspace/Deep_Insigth_Project/Zero_Shot/S3D_HowTo100M')
from s3dg import S3D


class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_lin=nn.Sequential(nn.Linear(512,256),
                                   nn.ReLU(),
                                   nn.Linear(256,128),
                                   nn.ReLU(),
                                   nn.Linear(128,64))
        

        self.text_lin=nn.Sequential(nn.Linear(512,256),
                                   nn.ReLU(),
                                   nn.Linear(256,128),
                                   nn.ReLU(),
                                   nn.Linear(128,64))


    def forward(self,img,text):
        return self.img_lin(img),self.text_lin(text)


class video_classifier(nn.Module):
    def __init__(self,num_cls=9):
        super().__init__()
        self.backbone=S3D('/home/jchen152/workspace/Deep_Insigth_Project/Zero_Shot/S3D_HowTo100M/s3d_dict.npy', 512)
        self.backbone.load_state_dict(torch.load('/home/jchen152/workspace/Deep_Insigth_Project/Zero_Shot/S3D_HowTo100M/s3d_howto100m.pth'))
        self.num_cls=num_cls

        self.classifier=nn.Sequential(nn.Linear(512,128),
                                      nn.ReLU(),
                                      nn.Linear(128,32),
                                      nn.ReLU(),
                                      nn.Linear(32,self.num_cls))

    def forward(self,x):
        x=self.backbone(x)['video_embedding']
        x=self.classifier(x)
        return x





if __name__=='__main__':
    img_feat=torch.randn((100,512))
    text_feat=torch.randn((100,512))
    net=classifier()
    a=1