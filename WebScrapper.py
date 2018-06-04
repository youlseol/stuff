# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:19:21 2017
@author: kueck
"""
# import libraries
import sys, re, os
import urllib3
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import gzip
import csv
import pickle
import json
import logging
 
class WebScrapper:
    '''
        web scrapper
    '''
    def __init__(self,download_image=False):
        
        self.download_image = download_image
        # specify the url
        self.domain = 'http://shop.hansalim.or.kr'
        self.root_dir = os.getcwd() + '\\hansalim\\' 
        self.img_size = 110,110
        self.pickle_file = 'dataset.pkl'
        self.save_file = os.path.join(self.root_dir,self.pickle_file) 
        self.network={}
        self.categories={}
        
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        
    def load_pickle(self):
        if not os.path.exists(self.save_file):
            self.save_pickle()
            
        print("Loading pickle file ...")
        with open(self.save_file, 'rb') as f:
            self.network = pickle.load(f)
        print("Done!")
    
    def save_pickle(self):
        
        if not os.path.exists(os.path.join(self.root_dir,'dataset.csv')):
            self.init_load()
         
        self.image_to_vector(useTorch=True)
        
        print("Creating pickle file ...")
        with open(self.save_file, 'wb') as f:
            pickle.dump(self.network, f)
        print("Done!")
    
    def init_load(self):
        if not os.path.exists(os.path.join(self.root_dir,'category.pickle')):
            self.download_category()
        else:
            self.load_category()
            
        dataset = []
        
        li = [(good,key) for key in self.categories
                   for good in self.categories[key]]
 
        
        for good,parent in li:
            url = self.domain + '/im/im/pd/IMPD0101.do?GDS_CLAS_CD={0}&clk='
            url = url.format(good)
            
            page = self.getPage(url)
            img_all, img_list = self.getImgfromPages(page)
            
            links = self.getInternalLinks(page)
            print('====== sub links =======')
            print(links)
            
            if len(links) > 0 :
                for link in links:
                    a,b = self.getImgfromPages(self.getPage(self.domain + link))
                    img_all = img_all + a
                    img_list = img_list + b
            
            pool = urllib3.PoolManager()
            for img_id, img_txt, img_path in img_list:
                if '_1_110.jpg' in img_id:
                    img_url = self.domain+img_path
                    
                    row = "({},{},{},{},{})".format(parent, good, img_id, img_txt, img_path)
                    print(row)
                    dataset.append((parent,good,img_id, img_txt, img_path))
                    
                    if self.download_image:
                        with pool.request('GET',img_url, preload_content=False, timeout=1, retries=True) as resp, open(os.path.join(self.root_dir,img_id), 'wb') as out_file:
                            data = resp.read()
                            out_file.write(data)
                
                        resp.release_conn()  
            
        sorted_dataset = sorted(set(dataset), key=lambda x:x[2])
        
#        with open(os.path.join(self.root_dir,'dataset.csv'),'w', encoding='utf-8') as out_csv_file:
#            csv_writer = csv.writer(out_csv_file, quotechar=',')
#            for f,t,_ in sorted_dataset:
#                csv_writer.writerow([f,t])
        
        import pandas as pd
        f, t, cidx, c = ([f for cidx,c,f,t,_ in sorted_dataset],[t for cidx,c,f,t,_ in sorted_dataset],[cidx for cidx,c,f,t,_ in sorted_dataset],[c for cidx,c,f,t,_ in sorted_dataset])
        pandas_dict =  {'class_id':cidx,'class':c,'image_name': f,'image_text': t }
        dataframe = pd.DataFrame(pandas_dict,columns=['image_name','image_text','class_id','class'])
        dataframe.to_csv(os.path.join(self.root_dir,'dataset.csv'),sep=',', encoding='utf-8')
        
        print('complete')
    
    def getPage(self, url):
        http_pool = urllib3.connection_from_url(url)
        r = http_pool.urlopen('GET',url)
        return r.data.decode('utf-8')
    
    def getInternalLinks(self, page):
        # parse the html using beautiful soap and store in variable `soup`
        soup = BeautifulSoup(page, 'html.parser')
        
        # Take out the <a> of name and get its value
        a_links = soup.find_all('a', attrs={'target':'productList'})
        links = [a.get('href') for a in a_links if a.get('class') is None]
        
        return links

        
    def getImgfromPages(self, page):
        # parse the html using beautiful soap and store in variable `soup`
        soup = BeautifulSoup(page, 'html.parser')
        
        # Take out the <img> of name and get its value
        img_all = soup.find_all('img')
        img_list = [(os.path.split(os.path.split(img.get('src'))[1])[1],
                    img.get('alt'),
                     img.get('src')
                     ) for img in img_all if '/im/is/itm/' in img.get('src')]
        
        return img_all, img_list
            
    def download_category(self):
        url = 'http://shop.hansalim.or.kr/im/main.do'
        http_pool = urllib3.connection_from_url(url)
        r = http_pool.urlopen('GET',url)
        page = r.data.decode('utf-8')
        # parse the html using beautiful soap and store in variable `soup`
        soup = BeautifulSoup(page, 'html.parser')
        # Take out the <div> of name and get its value
        img_all = soup.find_all('img',class_='second_category_menu_btn_img')
        second_category_list = [(img.get('id'),img.get('alt')) for img in img_all]
         
        for second_category in second_category_list:
            li_all = soup.find_all('li',class_='third_category_menu_btn',id=re.compile(second_category[0].replace('second','third').replace('img','menu')))
#            third_category_list = [(a.get('href')[-6:],a.string) for li in li_all
#                                                for a in li if a.string != '\n'
#                                                ]
            third_category_list = [a.get('href')[-6:] for li in li_all
                                                for a in li if a.string != '\n'
                                                ]
#            self.categories[(third_category_list[0][0][0:4],second_category[1])]=third_category_list 
            self.categories[third_category_list[0][0:4]]=third_category_list 
            
        print(self.categories)    
#        sys.setrecursionlimit(50000)    
        with open(os.path.join(self.root_dir,'category.pickle'),'wb') as out_file:
            # Pickle the 'data' dictionary using the highest protocol available.
            # object로 저장시 recursionError 발생, object -> str 
            pickle.dump(self.categories, out_file,-1)#pickle.HIGHEST_PROTOCOL)
            
        
        print('download complete..')
    
    def load_category(self):
        with open(os.path.join(self.root_dir,'category.pickle'),'rb') as in_file:
            self.categories = pickle.load(in_file)
   
        print(self.categories)
        print('load complete..')
    
    def _load_label(self, file_name):
        file_path = os.path.join(self.root_dir , file_name)
    
        print("Converting " + file_name + " to NumPy Array ...")
#        with gzip.open(file_path, 'rb') as f:
#                labels = np.frombuffer(f.read(), np.uint8, offset=8)
        
        with open(file_path, 'rb') as f:
                labels = np.frombuffer(f.read(), np.uint8, offset=8)
        print("Done")
    
        return labels
    
    
    def _load_img(self, file_name):
        file_path = os.path.join(self.root_dir, file_name)
        width, height = self.img_size
        print("Converting " + file_name + " to NumPy Array ...")
#        with gzip.open(file_path, 'rb') as f:
#                data = np.frombuffer(f.read(), np.uint8, offset=16)
#        with open(file_path, 'rb') as f:
#                data = np.frombuffer(f.read(), np.uint8, offset=16)
        with Image.open(file_path) as f:
            if f.size != self.img_size:
                f = f.resize(self.img_size)
            data = np.array(f)
        data = data.reshape(-1, width,height)
        print("Done")
    
        return data
    
    def _change_one_hot_label(self,X):
        T = np.zeros((X.size, 76))
        
        self.load_category()
        vl=[]
        for m in self.categories.values():
            vl+=m
            
        li = enumerate(vl) 
        d = {v: i for i,v in li} 
        for idx, row in enumerate(T):
            s='0'+str(X[idx])
            row[d[s]] = 1
        return T
    
    def image_to_vector(self, useTorch=False):
        print('build network ...')
#        files=[]
#        with open(os.path.join(self.root_dir,'dataset1.csv'),'r', encoding='utf-8') as csvfile:
#            read = csv.reader(csvfile, quotechar=',')
#            for row in read :
#                if len(row)>0:
#                    files.append(row)
         
        import pandas as pd
        dataframe = pd.read_csv(os.path.join(self.root_dir,'dataset.csv'))
        
        try:
            if useTorch:
                import torch
                li_img = [torch.FloatTensor(self._load_img(file)) for file in dataframe['image_name'].tolist()] 
                
                one_hot_label = self._change_one_hot_label(np.array(dataframe['class'].tolist()))
                li_label = [torch.LongTensor(classes) for classes in one_hot_label]  
                
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(li_img, li_label, random_state=0)
                
                self.network['train_img'] = torch.stack(X_train)
                self.network['test_img'] = torch.stack(X_test)
                self.network['train_label'] = torch.stack(y_train)
                self.network['test_label'] = torch.stack(y_test) 
                
            print('complete network ...')
        except FileNotFoundError as err: 
            print(err)
        
     
    def down_img(self,file_name):
        
        pool = urllib3.PoolManager()
        img_url =os.path.join('http://shop.hansalim.or.kr/im/is/itm/',file_name[:9]+'/'+file_name)
        down_dir = os.path.join(self.root_dir,file_name)
        print(img_url)
        with pool.request('GET',img_url, preload_content=False) as resp, open(down_dir, 'wb') as out_file:
            data = resp.read()
            out_file.write(data)
            print('download')
    
    
    def read_img(self, file_name):
        from skimage import io
        img_name = os.path.join(self.root_dir, file_name)
        image = io.imread(img_name)
        print(image)
    
    def update(self):
        self.load_pickle()
def main():
    webScrapper = WebScrapper(download_image=True)
    webScrapper.update();
   
        
if __name__ == '__main__':
    main() 