# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:31:29 2018
@author: kueck
"""
import os
from os.path import basename
import numpy as np
from skimage import io
from skimage.transform import rotate, resize, rescale
import cv2
class ImageTransformation:
    
    def __init__(self, root, save=False):
        self.root = root
        self.is_save = save
        
    def rescale1(self, image, name):
        ''' crop or !! after rescale
        '''
        for i in np.arange(0.5, 2, 0.3):
            m = rescale(image, i, mode='reflect')
            print('rescaled shape:')
            print(m.shape)
            if self.is_save:
                io.imsave(self.root +str(i)+'_'+ name,m)
            yield m
            
    def rotate1(self, image, name, angle=45):
        ''' background remove after rotate
        '''
        for i in range(1,8):
            m = rotate(image, angle*i, resize=True)
            print('rotated shape :')
            print(m.shape)
            if self.is_save:
                io.imsave(self.root+str(angle*i)+'_'+name,m)
            yield m
        
    def resize1(self, image, name, w=110,h=110):
        m = resize(image, (w,h), mode='reflect')
        print('resized shape:')
        print(m.shape)
        if self.is_save:
            io.imsave(self.root+str((w,h))+'_'+name,m)
        return m
    
    def translation(self, image, name):
        
        rows, cols = image.shape[:2]
        
        # 변환 행렬, X축으로 10, Y축으로 20 이동
        M = np.float32([[1,0,40],[0,1,30]])
        
        dst = cv2.warpAffine(image, M,(cols, rows))
#        cv2.imshow('Original', image)
#        cv2.imshow('Translation', image)
#        
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        io.imsave(self.root+'trans_'+name,image)
        
    def perspective(self, img, name):
        from matplotlib import pyplot as plt
        
#        img = cv2.imread('images/perspective.jpg') # 1600x2020 ==> 110 x 110 * (14 x 18)
        
        # [x,y] 좌표점을 4x2의 행렬로 작성
        # 좌표점은 좌상->좌하->우상->우하
        pts1 = np.float32([[32,5],[32,50],[83,5],[83,50]])
        
        # 좌표의 이동점
        pts2 = np.float32([[32,20],[10,110],[83,20],[110,110]])
        
        # pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
        cv2.circle(img, (32,5), 3, (255,0,0),-1)
        cv2.circle(img, (32,50), 3, (0,255,0),-1)
        cv2.circle(img, (83,5), 3, (0,0,255),-1)
        cv2.circle(img, (83,50), 3, (0,0,0),-1)
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        
        dst = cv2.warpPerspective(img, M, (110,110))
        
        plt.subplot(121),plt.imshow(img),plt.title('image')
        plt.subplot(122),plt.imshow(dst),plt.title('Perspective')
        plt.show()
        
        
        io.imsave(self.root+'trans_'+name,dst)
    
    def getFileName(self, file):
        return basename(file)
    
    def load(self):
        images = io.imread_collection(self.root+'*.jpg')
        
        i=1
        for (image, fn) in zip(images, images.files):
            name = self.getFileName(fn)
#            for m in self.rescale1(image, name):
#                for m1 in self.rotate1(m, name):
#                    io.imsave(self.root +str(i)+'_'+ name, self.resize1(m1, name))
#                    i+=1
#            self.translation(image,name)
            self.perspective(image,name)
def main():
    imageTransformation = ImageTransformation(os.getcwd()+'hansalim\\')
    imageTransformation.load()
if __name__ == '__main__':
    main()

