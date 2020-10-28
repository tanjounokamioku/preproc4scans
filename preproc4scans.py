#!/usr/bin/env python3

import numpy as np
import argparse
import cv2
from PIL import Image

# Routine to execute pre-processing of images from PDF files (scanned documents)
# To enhance OCR performance 

    # Load image
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to input image file")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    
    image = np.array(image)

    # Convert image to HSV profile
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # Estabilish highs and lows of what is considered "blue" (from signatures)
    blue_lo=np.array([110,50,50])
    blue_hi=np.array([130,255,255])

    # Mask those limits
    mask=cv2.inRange(hsv,blue_lo,blue_hi)

    # Remove blues (transform into white)
    image[mask>0]=(255,255,255)

    #cv2.imshow("color",image)
    #cv2.waitKey(0)
    
    # Convert to gray scale, change foreground to white and background to black with threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Get every non-zero pixel coordinates and use them to compute box-rotation
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    # 'cv2.minAreaRect' returns values between -90 and 0;
    # While rectangule rotates clockwise the angle gets close to 0,
    # In this case add 90 to angle
    if -4 < angle < -45:
        angle = -(90 + angle)
        
    There was a bug between -4 and 0 in which it rotated to the wrong way so if between them, do this
    elif angle <-4:
        angle = angle - angle
        
    # If positive, reverse angle
    else:
        angle = -angle
   
    # Convert again to gray scale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rotate 
    (h, w) = gray_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray_img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Show result
    print("[INFO] angle: {:.3f}".format(angle))
    #cv2.imshow("Rotated", rotated)
    #cv2.waitKey(0)
    
    # Reverse rotated image
    img = cv2.bitwise_not(rotated)
    #cv2.imshow("bit", img)
    th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    #cv2.imshow("th2", th2)
    #cv2.waitKey(0)
    horizontal = th2
    vertical = th2
    rows,cols = horizontal.shape

    # Separate horizontal lines
    horizontalsize = cols // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    #cv2.imshow("horizontal", horizontal)
    #cv2.waitKey(0)
    
    # Separate vertical lines
    verticalsize = rows // 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    #cv2.imshow("vertical", vertical)
    #cv2.waitKey(0)

    # Invert horizontals
    horizontal_inv = cv2.bitwise_not(horizontal)
    # Bitwise_and to mask horizontals
    masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
    # Go back to normal, line-less
    masked_img_inv = cv2.bitwise_not(masked_img)
    #cv2.imshow("sem linhas horizontais", masked_img_inv)
    #cv2.waitKey(0)

    # Invert verticals
    vertical_inv = cv2.bitwise_not(vertical)
    # Bitwise_and to mask verticals
    masked_img2 = cv2.bitwise_and(masked_img, masked_img, mask=vertical_inv)
    # Go back to normal, line-less
    masked_img_inv2 = cv2.bitwise_not(masked_img2)
    #cv2.imshow("sem linhas", masked_img_inv2)
    #cv2.waitKey(0)    
    
    image = np.array(masked_img_inv2)
    
    # Apply binary threshold to remove shadow from image
    ret, bin_img = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)
       
    #cv2.imshow("bin_img", bin_img)
    #cv2.waitKey(0)
    
    return bin_img
  



