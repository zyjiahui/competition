import os,sys,glob,shutil,json
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


# 自定义数据集 Dataset：对数据集的封装，提供索引方式的对数据样本进行读取
class SVHNDataset(Dataset):  
    # 初始化
    def __init__(self,img_path,img_label,transform=None):  # 参数：（图像，图像的标签，可选参数：图像的预处理）
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    # 获取特定索引的数据
    def __getitem__(self,index):
        img = Image.open(self.img_path[index]).convert('RGB')  # 将索引为index的图像转化为RGB格式的数据 

        if self.transform is not None:  # 如果需要对数据进行预处理
            img = self.transform(img)

        lbl = np.array(self.img_label[index],dtype=np.int32)  # 将标签数据加载为一个整数数组
        # lbl = list(lbl)  如果只是这样写，不会填充
        lbl = list(lbl) + (6-len(lbl)) * [10]  # 扩充这个数组  为了统一运算都变成定长的，不够6个的补10
        
        return img,torch.from_numpy(np.array(lbl[:6]))  # 返回的是索引为index的图像和标签

    # 数据集的长度
    def __len__(self):
        return len(self.img_path)


train_path = glob.glob('/Users/zhaoyuanjiahui/Deep_Learning/tianchi/Street_Character_Recognition/data/mchar_train/*.png')  # mchar_train文件中的所有图片
train_path.sort()  # 按字母顺序排列
train_json = json.load(open('/Users/zhaoyuanjiahui/Deep_Learning/tianchi/Street_Character_Recognition/data/train.json'))  # 载入json文件  路径容易写错
train_label = [train_json[x]['label'] for x in train_json]  # 在标签的json文件中，将标签label的内容提取出来，存储为列表list[]

# print(train_json['000000.png'])
# print([train_json[x]['label'] for x in train_json])




# 加入DataLoader的数据读取函数  DataLoader：对Dataset进行封装，提供批量读取的迭代读取
train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path,train_label,
                transforms.Compose([ # 数据扩增
                    transforms.Resize((64, 128)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    ),
    batch_size = 40, # 每批样本个数
    shuffle = False, # 是否打乱顺序
    # num_workers = 10 # 读取的线程个数
)




# 加入dataloader后，数据data的格式为  torch.Size([10,3,64,128]),torch.Size([10,6])


val_path = glob.glob('/Users/zhaoyuanjiahui/Deep_Learning/tianchi/Street_Character_Recognition/data/mchar_val/*.png')  # mchar_train文件中的所有图片
val_path.sort()  # 按字母顺序排列
val_json = json.load(open('/Users/zhaoyuanjiahui/Deep_Learning/tianchi/Street_Character_Recognition/data/val.json'))  # 载入json文件  路径容易写错
val_label = [val_json[x]['label'] for x in val_json]

# print(val_json['000000.png'])
val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path,val_label,
                transforms.Compose([ # 数据扩增
                    transforms.Resize((60, 120)),
                    # transforms.ColorJitter(0.3, 0.3, 0.2),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    ),
    batch_size = 40, # 每批样本个数
    shuffle = False, # 是否打乱顺序
    # num_workers = 10 # 读取的线程个数
)







print("终于跑通了")







