from dataprocess import train_loader,val_loader
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import numpy as np


# 定义模型
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1,self).__init__()
        # CNN特征提取模块
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=(3,3),stride=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,kernel_size=(3,3),stride=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 6个全连接层分类
        self.fc1 = nn.Linear(32*3*7,11)
        self.fc2 = nn.Linear(32*3*7,11)
        self.fc3 = nn.Linear(32*3*7,11)
        self.fc4 = nn.Linear(32*3*7,11)
        self.fc5 = nn.Linear(32*3*7,11)
        self.fc6 = nn.Linear(32*3*7,11)

    def forward(self,x):  
        feat = self.cnn(x)
        # 全连接之前需要将之前的数据拉平
        feat = feat.view(feat.shape[0],-1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1,c2,c3,c4,c5,c6



# 封装train训练操作
def train(train_loader,model,criterion,optimizer,epoch):
    model.train()
    train_loss = []
    for i,(data,target) in enumerate(train_loader):
        c0,c1,c2,c3,c4,c5 = model(data)
        loss = criterion(c0, target[:, 0]) + \
                criterion(c1, target[:, 1]) + \
                criterion(c2, target[:, 2]) + \
                criterion(c3, target[:, 3]) + \
                criterion(c4, target[:, 4]) + \
                criterion(c5, target[:, 5])
        # loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return np.mean(train_loss)


# 封装validate验证操作
def validate(val_loader,model,criterion):
    model.eval()
    val_loss = []

    with torch.no_grad():
        for i,(data,target) in enumerate(val_loader):
            c0, c1, c2, c3, c4, c5 = model(data)
            loss = criterion(c0, target[:, 0]) + \
                criterion(c1, target[:, 1]) + \
                criterion(c2, target[:, 2]) + \
                criterion(c3, target[:, 3]) + \
                criterion(c4, target[:, 4]) + \
                criterion(c5, target[:, 5])
            # loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)



model = SVHN_Model1()



# 训练与验证


lr = 0.001
epochs = 10
best_loss = 1000
# 定义损失
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(),lr=lr)


for epoch in range(epochs):
    print('Epoch: ',epoch)  # 这里输出了，后面卡住

    train_loss = train(train_loader,model,criterion,optimizer,epoch)  
    print('Train loss:{}'.format(train_loss))
    val_loss = validate(val_loader,model,criterion)
    print('Val loss:{}'.format(val_loss))

    # 记录验证集精度
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(),'./model.pt')  # 保存当前模型的参数，存储在model.pt中
    print('第{}轮跑完'.format(epoch))








# loss_plot = []  # 记录损失
# c0_plot = []  # 记录正确率
# 训练
# for epoch in range(epochs):
#     for data in train_loader:
#         c0,c1,c2,c3,c4,c5 = model(data[0])  # 6个位置的模型输出概率[p1，p2，p3，p4，p5，p6]
#         # 计算每一个字符的损失，并相加得处总损失，再求平均
#         # criterion参数（模型输出结果，真实标签），以此求出损失
#         loss = criterion(c0, data[1][:, 0]) + \
#                 criterion(c1, data[1][:, 1]) + \
#                 criterion(c2, data[1][:, 2]) + \
#                 criterion(c3, data[1][:, 3]) + \
#                 criterion(c4, data[1][:, 4]) + \
#                 criterion(c5, data[1][:, 5])
#         loss /= 6
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         loss_plot.append(loss.item())
#         c0_plot.append((c0.argmax(1) == data[1][:,0]).sum().item()*1.0 / c0.shape[0])
#     print(epoch)


