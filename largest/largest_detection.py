import cv2
import matplotlib.pyplot as plt
import numpy as np

image= cv2.imread('brightbullet.png', cv2.IMREAD_GRAYSCALE)
original_image= image

def Lobject():
    _, mask = cv2.threshold(image,220,255,cv2.THRESH_BINARY_INV)
    kernal = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(mask,kernal,iterations=4) 
    
    contours, hierarchy= cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image,contours,-1,(0,255,0),3)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)

    for (i,c) in enumerate(sorted_contours):
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, pts =[np.asarray(c)], color=(1))
        M = cv2.moments(mask,binaryImage = True)
        area = cv2.contourArea(c)
        cx= int(M['m10']/M['m00'])
        cy= int(M['m01']/M['m00'])
        cv2.putText(image, text= str(i+1), org=(cx,cy),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0),
                thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(image,str(area),(c[0,0,0],c[0,0,1]),2,0.8,(0,0,255),1)
    plt.imshow(image)
    plt.show()


def TDobject():
    pass


Lobject()
