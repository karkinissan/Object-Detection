import numpy as np
import cv2
from HelperMethods import *

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280, 720))
cap = cv2.VideoCapture(0)
sift = cv2.xfeatures2d.SIFT_create()
img1 = cv2.imread('logo.png', 0)
img1 = cv2.resize(img1,None,fx=1/2, fy=1/2, interpolation = cv2.INTER_AREA)
h, w = img1.shape
# img1.resize(512, 512)
kp1, des1 = sift.detectAndCompute(img1, None)
print(len(kp1),"Features in img1")
while (True):
    ret, frame = cap.read()
    if ret == True:
        frame = findLogo(frame, kp1=kp1, des1=des1, h=h, w=w)
        cv2.imshow("frame", frame)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
cap.release()
# out.release()
cv2.destroyAllWindows()
