# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:25:00 2018
@author: kueck
"""
import os
import pytesseract
#try: 
#    import Image 
#except ImportError: 
from PIL import Image 
import cv2
import numpy as np


"""
1.블러그 참고
http://emaru.tistory.com/15

2.download tesseract ocr exe and lnaguage pack
https://github.com/tesseract-ocr/tesseract/wiki
https://github.com/tesseract-ocr/langdata

3.reference
https://stackoverflow.com/questions/34225927/pytesseract-cannot-find-the-file-specified
Windows can't find the executable tesseract in the directories specified in your PATH environment variable. So either make sure that the directory containing tesseract is in your PATH variable or overwrite tesseract_cmd variable in your Python script like as following (put your PATH instead):
"""
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
# Example config: '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
# It's important to include double quotes around the dir path.

root = os.getcwd()
image = os.path.join(root,'car_number.jpg')#image1.png
print(image)
#text = pytesseract.image_to_string(Image.open(image), stdout="-l 'kor'", config=tessdata_dir_config)

img=cv2.imread(image,cv2.IMREAD_COLOR)
copy_img=img.copy()
img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg',img2)
blur = cv2.GaussianBlur(img2,(3,3),0)
cv2.imwrite('blur.jpg',blur)
canny=cv2.Canny(blur,100,200)
cv2.imwrite('canny.jpg',canny)


"""
https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html

mode parameter
CV_RETR_EXTERNAL retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
CV_RETR_LIST retrieves all of the contours without establishing any hierarchical relationships.
CV_RETR_CCOMP retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.
CV_RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours. This full hierarchy is built and shown in the OpenCV contours.c demo.

method parameter
CV_CHAIN_APPROX_NONE stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.
CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.
CV_CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS applies one of the flavors of the Teh-Chin chain approximation algorithm. See [TehChin89] for details.
"""
cnts,contours,hierarchy  = cv2.findContours(canny, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))

box1=[]
f_count=0
select=0
plate_width=0
  
for i in range(len(contours)):
     cnt=contours[i]          
     area = cv2.contourArea(cnt)
     x,y,w,h = cv2.boundingRect(cnt)
     rect_area=w*h  #area size
     aspect_ratio = float(w)/h # ratio = width/height
        
     if  (aspect_ratio>=0.2)and(aspect_ratio<=1.0)and(rect_area>=500)and(rect_area<=2000)and(w>=30)and(h>=40): 
          cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
          box1.append(cv2.boundingRect(cnt)) 


cv2.imwrite('snake.jpg',img)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


box1 = list(set(box1))       
for i in range(len(box1)): ##Buble Sort on python
     for j in range(len(box1)-(i+1)):
          if box1[j][0]>box1[j+1][0]:
               temp=box1[j]
               box1[j]=box1[j+1]
               box1[j+1]=temp
                 
##to find number plate measureing length between rectangles
#for m in range(len(box1)):
#     count=0
#     for n in range(m+1,(len(box1)-1)):
#          delta_x=abs(box1[n+1][0]-box1[m][0])
#          if delta_x > 150:
#               break
#          delta_y =abs(box1[n+1][1]-box1[m][1])
#          if delta_x ==0:
#               delta_x=1
#          if delta_y ==0:
#               delta_y=1           
#          gradient =float(delta_y) /float(delta_x)
#          if gradient<0.25:
#              count=count+1
#     #measure number plate size         
#     if count > f_count:
#          select = m
#          f_count = count;
#          plate_width=delta_x
          
#[(370, 331, 34, 43), (411, 332, 38, 43), (532, 332, 34, 44), (574, 332, 36, 44), (618, 331, 38, 43), (664, 329, 35, 44)]
number_plate=copy_img[box1[0][1]-10:box1[0][1]+box1[0][3]+10,box1[0][0]-10:box1[5][0]+box1[5][2]+10] #(y1:y2,x1:x2)
resize_plate=cv2.resize(number_plate,None,fx=1.8,fy=1.8,interpolation=cv2.INTER_CUBIC+cv2.INTER_LINEAR) 
plate_gray=cv2.cvtColor(resize_plate,cv2.COLOR_BGR2GRAY)
cv2.imwrite('plate_gray.jpg',plate_gray)

#ret,th_plate = cv2.threshold(plate_gray,150,255,cv2.THRESH_BINARY)
#cv2.imwrite('plate_th.jpg',th_plate)
#
#kernel = np.ones((3,3),np.uint8)
#er_plate = cv2.erode(th_plate,kernel,iterations=1)
#er_invplate = er_plate
#cv2.imwrite('er_plate.jpg',er_invplate)


#cv2.imshow('image1',plate_gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

text = pytesseract.image_to_string(plate_gray,lang='eng',config=tessdata_dir_config)

print('car number : '+text)
