import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

###自作データセット作成###
class Dataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data_info = list()
        for idx, data in enumerate(data):
            _data = torch.tensor(data)
            _data = _data.to(torch.float32)
            self.data_info.append((_data,label[idx]))
        
    def __getitem__(self, index):
        data, label = self.data_info[index] 
        return data, label
        
    def __len__(self):
        return len(self.data_info)

###ネットワークの定義###
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,2)

    def forward(self,x):
        z = self.fc1(x)
        return z
    
###AND gate###
#入力
and_data = []
and_data.append(([0,0]))
and_data.append(([0,1]))
and_data.append(([1,0]))
and_data.append(([1,1]))
#出力
label = [0,0,0,1]
##############

####XOR gate###
# xor_data = []
# xor_data.append(([?,?]))
# xor_data.append(([?,?]))
# xor_data.append(([?,?]))
# xor_data.append(([?,?]))
# label = [?,?,?,?]
################

###シード値設定###
torch.manual_seed(100)

###エポック数###
num_epochs = 50

###データセット作成###
dataset = Dataset(and_data,label)
dataloader = DataLoader(dataset,batch_size=1)

net = Net()

###損失関数###
criterion = nn.CrossEntropyLoss()

###最適化アルゴリズム###
optimizer = optim.SGD(net.parameters(), lr=0.01, 
                      momentum=0.9, weight_decay=5e-4)

for epoch in range(num_epochs):
    correct = 0
    total_loss = 0
    total = 0
    for i,(x,label) in enumerate(dataloader):
        ###ネットワークの出力###
        y_ = net(x)
        #最大値のインデックス取得
        z = torch.argmax(y_,dim=1)
        ###損失計算###
        loss = criterion(y_,label)
        
        ###勾配初期化###
        optimizer.zero_grad()
        ###誤差逆伝播###
        loss.backward()
        ###重み更新###
        optimizer.step()
        
        total_loss += loss.item()
        correct += (z == label).sum().item()
        total += x.shape[0]
        
        print('epoch:{}'.format(epoch),
              'loss:{:.3f}'.format(total_loss/(i+1)),
              'acc:{:.3f}'.format(correct/total*100))
    
    
    
