from sklearn.feature_extraction import img_to_graph
import streamlit as st
from soupsieve import select
import pickle
import pandas as pd
import requests
import base64
import joblib
import numpy as np
from pyexpat import model
import json
import cv2
import pywt
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img

class_dict = {'Cristiano Ronaldo': 0,
 'Dani Alves': 1,
 'Karim Benzema': 2,
 'Kevin De Bruyne': 3,
 'Lionel Messi': 4,
 'Luis Suarez': 5,
 'Luka Modric': 6,
 'Neymar Junior': 7,
 'Robert Lewandowski': 8,
 'Sergio Ramos': 9}

def load_saved_artifacts():
    #print("loading saved artifacts.....start")
    global __class_name_to_number
    global __class_number_to_name

    with open("C:\\Coding\\ML Practice\\Sports Celebrity\\server\\artifacts\\class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in class_dict.items()}

    global __model
   
    with open('C:\\Coding\\ML Practice\\Sports Celebrity\\server\\artifacts\\saved_model.pkl', 'rb') as f:
        __model = joblib.load(f)

    #print("loading saved artifacts....done")   

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

__class_name_to_number = {}
__class_number_to_name = {}
# __model = None 

def classify_image(img):
    imga = img
    imgs = get_cropped_image(img)
    result = []
    for img in imgs:
        scaled_raw_image = cv2.resize(img,(32,32))
        img_har = w2d(img,'db1',5)
        scaled_har_image = cv2.resize(img_har,(32,32))
        
        combined_img = np.vstack((scaled_raw_image.reshape(32*32*3,1),scaled_har_image.reshape(32*32*1,1)))
        len_image_array = 32*32*3+32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        class_dict = {}
        for i, name in zip(np.round(__model.predict_proba(final),2).tolist()[0],__class_name_to_number):
            class_dict[name] = i
        result.append((f'{class_number_to_name(__model.predict(final)[0])}'))
    if len(result) <= 1:
        return (f'Our Prediction: {class_number_to_name(__model.predict(final)[0])}')
    else:
    
        return (f'Our Predictions: {result}')

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def get_cropped_image(img):
    face_cascade = cv2.CascadeClassifier('C:\\Coding\\ML Practice\\Sports Celebrity\\server\\opencv\\haarcascades\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:\\Coding\\ML Practice\\Sports Celebrity\\server\\opencv\\haarcascades\\haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces                
  


st.title("Football Recognizer System")
st.subheader("Put in an image and we will find the most similar looking face")

img_data = st.file_uploader(label = 'load image for recognition', type = ['jpg','png','jpeg'])

if img_data is not None:
    uploaded_img = Image.open(img_data)
    st.image(uploaded_img)
    img_path = f'C:\\Coding\\ML Practice\\Sports Celebrity\\server\\test_images\\{img_data.name}'
    # img_path = load_img(img_path)
    load_saved_artifacts()
    img = cv2.imread(str(img_path))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # plt.axis('off')
    # plt.title('Your Input Image')
    # plt.imshow(gray, cmap='gray')
    ans = classify_image(img)
    st.success(ans)   

