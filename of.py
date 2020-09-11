import cv2 as cv
import numpy as np 
import time
from numpy import save
from numpy import asarray


cap = cv.VideoCapture('/home/mostafathereal/Desktop/speed_challenge/data/test.mp4')

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


for i in range(0, 20390):

    ret, frame = cap.read()
    cv.imshow('test input', frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    
    # Calculates dense qoptical flow by Farneback method 
    # Computes the magnitude and angle of the 2D vectors
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  
    
    # angle in radians
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1]) 
    print(type(min(angle[0])))

    # print(len(magnitude), len(magnitude[0]))
    # print(len(angle), len(angle[0]))
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