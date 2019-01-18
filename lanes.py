import cv2
import matplotlib.pyplot as pyplot
import numpy as np

def region_of_interest(image):
    #identifying region of interest from image

    height = image.shape[0]  #extracting eight from original shape of image
    polygonals = np.array([  
        [(200, height), (1100, height), (550,25)]
    ])
    mask = np.zeros_like(image)         #making black image using np zeros image 
    cv2.fillPoly(mask,polygonals,255)  #drawing poygons 

    masked_image = cv2.bitwise_and(image, mask)   #bit wise ADN of image and mask
    return masked_image

def canny(image):
    #this is the function to detect edge from image
    #converting image to gray 
    gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)  
    blour = cv2.GaussianBlur(gray, (5,5),0)   #making image blour

    #detecting edge using cv2 Canny algotithm
    canny = cv2.Canny(blour, 50, 150)
    return canny


#importing image 
image = cv2.imread("lane.jpg")
lane_image = np.copy(image)  #copying lane image to make change in copy image

canny = canny(lane_image)   #calling canny eade detection function
cropped_image = region_of_interest(canny)

cv2.imshow("cropped image ", cropped_image)
cv2.waitKey(0)
