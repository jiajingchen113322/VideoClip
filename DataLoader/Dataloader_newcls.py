import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys
from torch.utils.data import DataLoader
import cv2
import torchvision.transforms as transforms
import json


class Frame_Dataset(Dataset):
    def __init__(self,root,mode='train',fold=0,video_based=True):
        super().__init__()
        self.root=root
        self.fold_split_json='/home/jchen152/workspace/Deep_Insigth_Project/Zero_Shot/others/sub_split_profile.json'
        self.fold=fold
        # self.ignore_label=np.arange(8,18)
        self.data_dict=self.create_data() # {label:{personID:[img]}}    
        self.total_person_list=np.array(self.get_person_ID_list())
        self.train_person,self.test_person= self.get_train_test_person()
        
        # self.train_person=['76189','61597','25470','79336','56306','65818','49381','76803','24491','19332']
        # self.test_person=['24026','42271']
        
        self.video_length=30
        self.video_based=video_based
        # ===== get train or test person list ========
        # the last one third person will be used for testing
        person_id_index=np.zeros(len(self.total_person_list))
        # person_id_index[self.fold*5:(self.fold+1)*5]=1
        
        # testing_numer=len(self.total_person_list)//3
        if mode=='train':
            
            # person_id_index=1-person_id_index
            # person_id_index=person_id_index.astype(np.bool8)
            # self.person_list=self.total_person_list[person_id_index]
            for i in self.train_person:
                index=np.where(self.total_person_list==i)[0][0]
                person_id_index[index]=1
            person_id_index=person_id_index.astype(np.bool8)
            self.person_list=self.total_person_list[person_id_index]
            
            
        else:
            # person_id_index=person_id_index.astype(np.bool8)
            # self.person_list=self.total_person_list[person_id_index]
            for i in self.test_person:
                index=np.where(self.total_person_list==i)[0][0]
                person_id_index[index]=1
            person_id_index=person_id_index.astype(np.bool8)
            self.person_list=self.total_person_list[person_id_index]
            
        # =============================================
        
        self.frame_list,self.label_list=self.get_frame_list()

        self.transforms=transforms.Normalize(mean=[0.481, 0.456, 0.408], std=[0.268, 0.261, 0.275])


    def get_train_test_person(self):
        j_file=open(self.fold_split_json)
        j_file=json.load(j_file)
        fold_keys=list(j_file.keys())
        split_info=j_file[fold_keys[self.fold]]
        
        train_person_list=split_info['train']
        test_person_list=split_info['test'] 
        return train_person_list,test_person_list   
    
    
    
    

    def get_frame_list(self):
        frame_list=[]
        label_list=[]
        
        for label in list(self.data_dict.keys()):
            for person in self.person_list:
                if not person in list(self.data_dict[label].keys()):
                    continue
                
                
                frame_list+=self.data_dict[label][person]
                label_list+=[int(label)]*len(self.data_dict[label][person])
        
        return frame_list,label_list

        
    def create_data(self):
        data_dict={}
        for i in range(8):
            # if i in self.ignore_label:
            #     continue
            if i not in list((data_dict.keys())):
                data_dict[i]={}
        
        label_set=os.listdir(self.root)
        for label in label_set:
            # if int(label) in self.ignore_label:
            #     continue
            img_fold_path=os.path.join(self.root,label)
            img_list=os.listdir(img_fold_path)
            for img in img_list:
                img_name,img_format=os.path.splitext(img)
                if img_format!='.jpg':
                    continue
                
                split_name=img_name.strip().split('_')
                
                img_view=split_name[0]
                if img_view not in ['Dashboard']:
                    continue
                
                person_ID=split_name[1]
                
                if person_ID not in list(data_dict[int(label)].keys()):
                    data_dict[int(label)][person_ID]=[]
                
                data_dict[int(label)][person_ID].append(os.path.join(img_fold_path,img))
    

        return data_dict
    
    def get_person_ID_list(self):
        person_id_list=set(self.data_dict[0].keys())
        return sorted(person_id_list)

    def get_video(self,frame_path,label):
        img_name=os.path.split(frame_path)[-1]
        img_name=os.path.splitext(img_name)[0]
        split_img_name=img_name.strip().split('_')
        
        person_ID=split_img_name[1]
        person_frame_pool=self.data_dict[label][person_ID]
        
        def get_frame_num(x):
            image_name=os.path.split(x)[-1]
            image_name=os.path.splitext(image_name)[0]
            split_image_name=image_name.strip().split('_')
            frame_index=split_image_name.index('no')+1
            frame_num=int(split_image_name[frame_index])
            return frame_num

        
        
        person_frame_pool=sorted(person_frame_pool,key=lambda x:get_frame_num(x))
        
        path_index=person_frame_pool.index(frame_path)
        video_start=np.max([path_index-self.video_length//2,0])
        video_end=np.min([path_index+self.video_length//2,len(person_frame_pool)])
        video_list=person_frame_pool[video_start:video_end]
        if len(video_list)<self.video_length:
            video_list+=[video_list[-1]]*(self.video_length-len(video_list))
        
        video_data=[torch.FloatTensor(cv2.imread(i)) for i in video_list]
        video_data=torch.stack(video_data,0)
        return video_data
        
    def __getitem__(self,index):
        frame_path=self.frame_list[index]
        label=self.label_list[index]
        
        if self.video_based:
            video_data=self.get_video(frame_path,label)
            video_data=video_data.permute(3,0,1,2)
            return video_data,label,frame_path

        else:
            img=torch.FloatTensor(cv2.imread(frame_path))
            img=img.permute(2,0,1)
            img=img/255
            return img,label


    def __len__(self):
        # return 10
        return len(self.frame_list)



def get_sets(data_path,batch_size=10,fold=0,video_based=True):
    train_dataset=Frame_Dataset(root=data_path,mode='train',fold=fold,video_based=video_based)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    val_dataset=Frame_Dataset(root=data_path,mode='test',fold=fold,video_based=video_based)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    
    return train_loader,val_loader





if __name__=='__main__':
    # json_file_path='/data1/jiajing/worksapce/Project/Zero_Shot/others/sub_split_profile.json'
    # j_file=open(json_file_path)
    # j_file=json.load(j_file)
    # a=1
    
    
    root='/data1/jiajing/worksapce/Project/Zero_Shot/Data/syn5fps'
    train_loader,test_loader=get_sets(data_path=root,video_based=True)
    
    for i in test_loader:
        a=1
    
    dataset=Frame_Dataset(root=root)
    for i in dataset:
        continue