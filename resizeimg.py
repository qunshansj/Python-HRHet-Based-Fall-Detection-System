
import cv2
from numpy import *
import numpy as np

def resizeimg(img):
    height, width, channels = img.shape
    if width > 1500 or width < 600:
        scale = 1200 / width
        #print("图片的尺寸由 %dx%d, 调整到 %dx%d" % (width, height, width * scale, height * scale))
        scaled = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    else:
        scaled = img
    return scaled

normal_file = './train3'
cap = cv2.VideoCapture("./3.mp4")
num = 0
num2 = 0
while(1):
    try:
        ret, frame = cap.read()
        num += 1
        if num == 10:
            num = 0
            num2 += 1
            cv2.imwrite(normal_file + "/" + str(num2) + '.jpg',frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    except:
        break
