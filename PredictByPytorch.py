# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:14:22 2018
@author: kueck
"""
import os
from TrainByPytorch import *
from TrainByPytorch import Net,Variable,torch
from skimage import io 

net = Net()
net.load_model()
root_dir = os.getcwd()
img_path = os.path.join(root_dir, 'hansalim\\010101018_1_110.jpg')
images = io.imread(img_path)
images = images.reshape(-1,110,110)
timg = torch.FloatTensor(images)
simg = torch.stack([timg])
image = Variable(simg)
outputs = net(image)
_, predicted = torch.max(outputs.data, 1)
 
classes = ['멥쌀/찹쌀','잡곡/깨'    ,'콩류'    ,'혼합곡/보리/유색미'    ,'콩나물/숙주'    ,'잎/줄기채소'    ,'열매채소'    ,'뿌리채소'    ,'버섯류'    ,'말린나물류'    ,'손질한채소'    ,'양념/조미채소'    ,'건조/냉동과일'    ,'견과류'    ,'과일/과채'    ,'유정란'    ,'한우'    ,'돼지'    ,'닭/오리'    ,'햄/소시지'    ,'육가공'    ,'생선/손질생선'    ,'멸치/황태/건어'    ,'패류/오징어/새우'    ,'미역/다시마/해조'    ,'김/조미김'    ,'젓갈/게장'    ,'말린오징어/어포'    ,'두부/어묵/묵'    ,'김치/밑반찬'    ,'된장/고추장/간장'    ,'기름/식초/소금'    ,'소스/양념/조청'    ,'곡식가루/혼합가루'    ,'과채/고춧가루'    ,'라면/간편조리면'    ,'국수/면'    ,'만두/피자/핫도그'    ,'씨리얼/생식/선식'    ,'국/탕/요리'    ,'죽/간편밥/밥양념'    ,'우리밀빵'    ,'떡/한과'    ,'과자'    ,'우유/두유/유제품'    ,'과즙/발효/전통/음용식초'    ,'잼/푸딩/빙과'    ,'홍삼/녹용/산양삼'    ,'건강즙/농축액'    ,'분말/환/절편'    ,'꿀/화분/로얄젤리'    ,'건강차'    ,'일반차'    ,'기초화장'    ,'색조화장품'    ,'세안제/팩'    ,'유아/썬크림'    ,'바디/핸드'    ,'남성화장품'    ,'오가닉코튼'    ,'주방세제/세정'    ,'세제/세탁비누'    ,'샴푸/린스/헤어용품'    ,'세안비누/온몸세정'    ,'치약/칫솔'    ,'주방소품'    ,'휴지/티슈'    ,'도서'    ,'생리대'    ,'숯/원예'    ,'위생/환경용품'    ,'생활소품'    ,'선물포장'    ,'천연벽지'    ,'기금'    ,'대용량/어린이집']

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread(img_path)
imgplot = plt.imshow(img)
plt.show()
print('predicted :'+classes[int(predicted)])