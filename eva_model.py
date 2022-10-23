import os
import sys
sys.path.append('./S3D_HowTo100M')

from s3dg import S3D
import numpy as np
import torch
from DataLoader.DataLoader import get_sets
from tqdm import tqdm
from util.get_acc import cal_cfm
# ===== configuration =====
data_path='/data1/jiajing/worksapce/Project/Zero_Shot/Data/synDD_1fps'
device='cuda'
bs=4
# ========================


# ======== get model =======
model=S3D('/data1/jiajing/worksapce/Project/Zero_Shot/S3D_HowTo100M/s3d_dict.npy', 512)
# model.load_state_dict(torch.load('S3D_HowTo100M\s3d_howto100m.pth'))
model.load_state_dict(torch.load('/data1/jiajing/worksapce/Project/Zero_Shot/Exp/training_lr_3*1e-4/pth_file/epoch_34')['model_state'])
model.eval()
model=model.to(device)
# ==========================

# ======= create txt ==========
label_map = {
	0:	"Normal Driving",   # *
	1:	"Drinking",                 # *
	2:	"Phone Call right",        # *
	3:	"Phone Call left",         # *
	4:	"Eating",
	5:	"Text light",
	6:	"Text reft",
	7:	"Hair or makeup",
	8:	"Reaching behind",
	9:	"Adjust control panel",
	10:	"Pick up from floor (Driver)",
	11:	"Pick up from floor (Passenger)",
	12:	"Talk to passenger at the right",
	13:	"Talk to passenger at backseat",
	14:	"yawning",
	15:	"Hand on head",
	16:	"Singing with music",
	17:	"shaking or dancing with music",
}
text_list=list(label_map.values())[:8]
# =================================

train_loader, test_loader=get_sets(data_path=data_path,batch_size=bs)

total_cf=np.zeros((len(text_list),len(text_list)))
for (x,y) in tqdm(test_loader):
    with torch.no_grad():
        x,y=x.to(device),y.to(device)
        video_output=model(x)
        text_output=model.text_module(text_list)
        video_embedding = video_output['video_embedding']
        text_embedding = text_output['text_embedding']
        similarity_matrix = torch.softmax(torch.matmul(text_embedding, video_embedding.t()).t(),-1)
        # pred=torch.argmax(similarity_matrix,-1)
        batch_cfm=cal_cfm(similarity_matrix,y,ncls=len(text_list))
        total_cf+=batch_cfm
a=1
