# -*- coding: utf-8 -*-

import os
import torch
#import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms, utils 
import pickle
from PIL import Image

from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F


model_dir = os.getcwd()
model_name = '\\model_by_pytorch.pkl'

classes = ['멥쌀/찹쌀'
,'잡곡/깨'
,'콩류'
,'혼합곡/보리/유색미'
,'콩나물/숙주'
,'잎/줄기채소'
,'열매채소'
,'뿌리채소'
,'버섯류'
,'말린나물류'
,'손질한채소'
,'양념/조미채소'
,'건조/냉동과일'
,'견과류'
,'과일/과채'
,'유정란'
,'한우'
,'돼지'
,'닭/오리'
,'햄/소시지'
,'육가공'
,'생선/손질생선'
,'멸치/황태/건어'
,'패류/오징어/새우'
,'미역/다시마/해조'
,'김/조미김'
,'젓갈/게장'
,'말린오징어/어포'
,'두부/어묵/묵'
,'김치/밑반찬'
,'된장/고추장/간장'
,'기름/식초/소금'
,'소스/양념/조청'
,'곡식가루/혼합가루'
,'과채/고춧가루'
,'라면/간편조리면'
,'국수/면'
,'만두/피자/핫도그'
,'씨리얼/생식/선식'
,'국/탕/요리'
,'죽/간편밥/밥양념'
,'우리밀빵'
,'떡/한과'
,'과자'
,'우유/두유/유제품'
,'과즙/발효/전통/음용식초'
,'잼/푸딩/빙과'
,'홍삼/녹용/산양삼'
,'건강즙/농축액'
,'분말/환/절편'
,'꿀/화분/로얄젤리'
,'건강차'
,'일반차'
,'기초화장'
,'색조화장품'
,'세안제/팩'
,'유아/썬크림'
,'바디/핸드'
,'남성화장품'
,'오가닉코튼'
,'주방세제/세정'
,'세제/세탁비누'
,'샴푸/린스/헤어용품'
,'세안비누/온몸세정'
,'치약/칫솔'
,'주방소품'
,'휴지/티슈'
,'도서'
,'생리대'
,'숯/원예'
,'위생/환경용품'
,'생활소품'
,'선물포장'
,'천연벽지'
,'기금'
,'대용량/어린이집']

class ExDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        transform :
    """

    def __init__(self, data_tensor, target_tensor, transform):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data_tensor[index]), self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
    
class Net(nn.Module):
    
    root_dir = os.getcwd()+ '\\hansalim\\'
        
    '''
        CNN
        Conv-Relu-(Pooling) -> Conv-Relu-(Pooling) -> Affine(full-connected)-Relu -> Affine-softmax(loss)
        
        Conv * filter(weight) * Relu + bias = out 
        
        [Convolution layer output size calculater]
        input (H, w), filter (FH, FW), output (OH, OW), padding P, stride S 
        OH = (H + 2P -FH)/S + 1
        OW = (W + 2P -FW)/S + 1
        [Pooling layer output size calculater]
        OH = H/P
        OW = W/P
    '''
    def __init__(self, num_classes=76):
        super(Net, self).__init__()
        #들어가기 앞서 정규화 과정 거치면 속도 개선
         
        # 3 input image channel, 16 output channels, 5x5 square convolution
        # kernel == filter == weight
         #보통 stride 와 padding은 같은 값 적용 
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, padding=2),  # input 110*110*3 -> 110*110*16 
                                    nn.BatchNorm2d(16), 
                                    nn.ReLU(), 
                                    nn.MaxPool2d(2))                             # Max pooling over a (2, 2) window, 55*55*16
        
        # If the size is a square you can only specify a single number
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, padding=2), # input 55*55*16 -> 55*55*32
                                    nn.BatchNorm2d(32), 
                                    nn.ReLU(), 
                                    nn.MaxPool2d(2))                             # Max pooling over a (2, 2) window, 55*55*32 -> 27*27*32
        
#        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=2), # input 27*27*32 -> 27*27*64  -> 27.5 소실
#                                    nn.BatchNorm2d(64), 
#                                    nn.ReLU(), 
#                                    nn.MaxPool2d(2))                             # Max pooling over a (2, 2) window, 27*27*64 -> 13*13*64
        
        # an affine operation: y = Wx + b 
        # 기하학 affine : 행렬의 내적을 구한다.
        # 마지막은 출력값 지정 (76)
        self.fc = nn.Linear(27*27*32, 76)  # affine1 -full-connected 
#        
        
#        self.features = nn.Sequential(
#            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=3, stride=2),
#            
#            nn.Conv2d(64, 192, kernel_size=5, padding=2),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=3, stride=2),
#            
#            nn.Conv2d(192, 384, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(384, 256, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(256, 256, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=3, stride=2),
#        )
#        self.classifier = nn.Sequential(
#            nn.Dropout(),
#            nn.Linear(256 * 2 * 2, 4096),
##            nn.Linear(256 * 6 * 6, 4096),
#            nn.ReLU(inplace=True),
#            nn.Dropout(),
#            nn.Linear(4096, 4096),
#            nn.ReLU(inplace=True),
#            nn.Linear(4096, num_classes),
#        )
         

        print(">>>> Loading pickle file ...")
        with open(os.path.join(self.root_dir,'dataset.pkl'), 'rb') as f:
            self.network = pickle.load(f)
        print(">>>> Done!")
         
    def forward(self, x): 
        
        x = self.layer1(x) 
        x = self.layer2(x)  
        x = x.view(x.size(0), -1) # 성형,shape 변환 = im2col
        x = self.fc(x) 
        
#        x = self.features(x)
##        print(x.shape)
#        x = x.view(x.size(0), 256 * 2 * 2)#256 * 6 * 6)
#        x = self.classifier(x)
  
        return x
    
    def num_flat_features(self, x):
        '''
        tensor to 2d
        '''
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
    def get_output_size(self,wh,p,fwh,s):
        return (wh + 2*p -fwh)/s + 1
    
    def train_model(self, model, optimizer, criterion, num_epoch = 10, use_gpu = False):
        
        if use_gpu:
            model = model.cuda() 
        
        print(model)
         
#        #transform
#        normalize = transforms.Normalize(
#           mean=[0.485, 0.456, 0.406],
#           std=[0.229, 0.224, 0.225]
#        )
#        ds_trans = transforms.Compose([transforms.Scale(224),
#                                       transforms.CenterCrop(224),
#                                       transforms.ToTensor(),
#                                       normalize])
        
        dataset = TensorDataset(self.network['train_img'],self.network['train_label'])
#        dataset = ExDataset(self.network['train_img'],self.network['train_label'],ds_trans)
        dataloader = DataLoader(dataset, batch_size=4,shuffle=False, num_workers=0)
        
        loss_list = []
        
        print('>>>> start Training')
        for epoch in range(num_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0): 
                
                # get the inputs
                inputs, labels = data   
                
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
 
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = model(inputs) 
#                print(outputs.data)
#                print('>>>> calculating loss...')
#                loss = criterion(outputs, labels)
                
#                print(torch.max(labels, 1)[1])
                loss = criterion(outputs, torch.max(labels, 1)[1])
#                print('>>>> loss: %.3f' % loss)
                
                loss.backward()
                
                optimizer.step()
        
                # print statistics
                running_loss += loss.data[0]
       
                print('>>>> [%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 5))
                
                if i % 10 == 0:    # print every 5 mini-batches
                    loss_list.append(running_loss / 5)
                
                running_loss = 0.0
        
        print('>>>> Finished Training')
        
        lx = np.arange(len(loss_list))
 
        plt.plot(lx, loss_list, label='train loss') 
        plt.grid()
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('error rate')
        plt.title('Deep learning') 
        plt.show()
                
        torch.save(self.state_dict(),model_dir+model_name)
        
    def test_model(self, model, use_gpu):
        
        self.load_model()
        
        dataset = TensorDataset(self.network['test_img'],self.network['test_label'])
        dataloader = DataLoader(dataset, batch_size=4,shuffle=True, num_workers=0)
             
        
        dataiter = iter(dataloader)
        images, labels = dataiter.next() 
              
        # functions to show an image
        def imshow(images):
            npimg = lambda img : img.cpu().numpy().reshape(110,110,-1).astype(int)
             
            imgs = [npimg(m) for m in images.data]
            fig=plt.figure(figsize=(8, 8))
            columns = len(images)
            rows = 1
            for i in range(1, columns*rows +1):
                fig.add_subplot(rows, columns, i)
                plt.imshow(imgs[i-1])
            plt.show()
                
        correct = 0
        total = 0
        for images, labels in dataloader:
            
            if use_gpu: 
                model = model.cuda()
                images, labels = Variable(images.cuda()), Variable(labels.cuda())   
            else:
                images, labels = Variable(images), Variable(labels)
            
            # show images
            imshow(images) 
        
            outputs = model(images)
#            print('4-outputs data : ', outputs.data)
            
            _, predicted = torch.max(outputs.data, 1)
#            print('4-predicted : ',predicted)
            
            total += labels.size(0)
            labels = torch.max(labels,1)[1]
#            print('4-labels : ', labels)
            
            correct += (predicted == labels.data).sum() 
            
            print('GroundTruth: ', ' '.join('%5s' % classes[labels.data[j]] for j in range(len(labels)))) 
            print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(predicted))))
             
        print('Test Accuracy of the model on the {0} test images: {1:.3f}'.format(total, (100 * correct / total)))
        
#        #정확도 90프로 이하면 model parameter 파일 삭제
#        import math
#        if math.ceil(100 * correct / total) < 90:
#            os.remove(model_dir+model_name)
    
    def load_model(self):
        self.load_state_dict(torch.load(model_dir+model_name))
        
def main(device):         
    net = Net()
#    print(net)
    
#    params = list(net.parameters())
#    #print(params[0].size())  # conv1's .weight
#    for p in params:
#        print(p.size())
    
    
    from torchvision import models
    use_gpu = torch.cuda.is_available()
    
#    model = models.vgg11(pretrained=False, num_classes=76) 
    model = net
    
#    use_parallel = True
#    if use_parallel:
#        print("[Using all the available GPUs]")
#        model = nn.DataParallel(model, device_ids=[0, 1])
        
#    ## Freezing all layers
#    for params in model.parameters():
#        params.requires_grad = False
#        
#    ## Freezing the first few layers. Here I am freezing the first 7 layers 
#    ct = 0
#    for name, child in model.named_children():
#        ct += 1
#        if ct < 7:
#            for name2, params in child.named_parameters():
#                params.requires_grad = False
            
  
    import torch.optim as optim
    #분류에선 출력함수(softmax)에 교차엔트로피오류 적용
    criterion = nn.CrossEntropyLoss()
    # 가중치초깃값 개선 함(Adagrad 단점 보안 : Adadelta(over fitting)/RMSprop(처음에 학습 다되고 더이상 안됨,빠르고좋음))
    optimizer = optim.RMSprop(net.parameters(), lr=0.001)
#    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    
     
    if not os.path.exists(model_dir+model_name):
        net.train_model(model, optimizer, criterion, 10, use_gpu)
    else:
        net.test_model(model, use_gpu)
    
        

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()
    args.device = None
#    if not args.disable_cuda and torch.cuda.is_available():
#        args.device = torch.device('cuda')
#    else:
#        args.device = torch.device('cpu')
#        
    main(args.device)