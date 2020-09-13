import cv2 as cv
import numpy as np 
import time
from numpy import save, loadtxt
import torch

labels = loadtxt('/home/mostafathereal/Desktop/comma.ai_speed_challenge/data/train.txt', delimiter = "\n")

cap = cv.VideoCapture('/home/mostafathereal/Desktop/comma.ai_speed_challenge/data/train.mp4')

width  = cap.get(3)
height = cap.get(4)
print("WIDTH -- ", width, "HEIGHT -- ", height)

# ret = a boolean return value from 
# getting the frame, first_frame = the 
# first frame in the entire video sequence 
ret, first_frame = cap.read() 
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY) 
  
# Creates an image filled with zero 
# intensities with the same dimensions  
# as the frame 
mask = np.zeros_like(first_frame) 
  
# Sets image saturation to maximum 
mask[..., 1] = 255

train_data = []
label_data = []

for i in range(1, 20397):

        ret, frame = cap.read()
        cv.imshow('train set', frame)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 

        # Calculates dense qoptical flow by Farneback method 
        # Computes the magnitude and angle of the 2D vectors
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  

        # angle in radians
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1]) 


        # print(len(magnitude), len(magnitude[0]))
        # print(len(angle), len(angle[0]))
        # print(len(gray), len(gray[0]))
        arr = [[magnitude, angle, gray/255]]
        train_data.append(arr)
        label_data.append(labels[i])
        print("saved frame ", i)

        # print("yeehaw")q
        # Converts HSV to RGB (BGR) color representation 
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX) 

        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR) 

        # Opens a new window and displays the output frame 
        cv.imshow("dense optical flow", rgb) 

        # Updates previous frame 
        prev_gray = gray
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv.destroyAllWindows()

# print(type(train_data))
# print(type(label_data))
print("ye")
torch.save(torch.HalfTensor(train_data), '/home/mostafathereal/Desktop/comma.ai_speed_challenge/data1/train.pt')
print("ye1")
torch.save(torch.HalfTensor(label_data), '/home/mostafathereal/Desktop/comma.ai_speed_challenge/data1/train_labels.pt')
print("ye3")

