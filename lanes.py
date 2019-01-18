import cv2
import matplotlib.pyplot as pyplot
import numpy as np

def region_of_interest(image):
    #identifying region of interest from image

    height = image.shape[0]  #extracting eight from original shape of image
    polygonals = np.array([  
        [(200, height), (1100, height), (550,250)]
    ])
    mask = np.zeros_like(image)         #making black image using np zeros image 
    cv2.fillPoly(mask,polygonals,255)  #drawing poygons 

    masked_image = cv2.bitwise_and(image, mask)   #bit wise ADN of image and mask
    return masked_image

def display_lines(img,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image




def canny(image):
    #this is the function to detect edge from image
    #converting image to gray 
    gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)  
    blour = cv2.GaussianBlur(gray, (5,5),0)   #making image blour

    #detecting edge using cv2 Canny algotithm
    canny = cv2.Canny(blour, 50, 150)
    return canny


#importing image 
image = cv2.imread("images/lane.jpg")
lane_image = np.copy(image)  #copying lane image to make change in copy image

canny = canny(lane_image)   #calling canny eade detection function
cropped_image = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,np.array([]),minLineLength=40, maxLineGap=5)

#linned image 
line_image = display_lines(lane_image,lines)

#image and lines combined
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) 


cv2.imshow("Linned image", combo_image)  #line image
cv2.waitKey(0)
