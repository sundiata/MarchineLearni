import numpy as np
import pywt
import cv2

def w2d(img, mode='haar', level = 1):
    img = np.array(img, dtype=np.uint8)
    imArray = img
    # Datatype conversion
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
    # convert to float
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    
    #reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    
    return imArray_H