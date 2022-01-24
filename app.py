import streamlit as st
import tensorflow as tf
import keras 
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import convolutional
from keras.layers import pooling
from keras.layers import core
from keras import optimizers
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.models import load_model
from bs4 import BeautifulSoup
import urllib.request
import os
import requests
from os.path import basename
import shutil
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt

#model_3=Sequential()
model_8 = Sequential()
model_8.add(convolutional.Convolution2D(32, (9,9),strides=(3, 3), input_shape=(512,512,1)))
model_8.add(Activation('relu'))
model_8.add(pooling.MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model_8.add(convolutional.Convolution2D(64, (2,2),strides=(2, 2)))
model_8.add(Activation('relu'))
model_8.add(pooling.MaxPooling2D(pool_size=(3, 3),strides=(2, 2)))
model_8.add(convolutional.Convolution2D(128, (2,2),strides=(2, 2)))
model_8.add(BatchNormalization())
model_8.add(Activation('relu'))
model_8.add(pooling.MaxPooling2D(pool_size=(3, 3),strides=(1, 1)))
model_8.add(Flatten())
model_8.add(Dense(4096,activation='relu'))
model_8.add(core.Dropout(.3))
model_8.add(Dense(1,activation='linear'))
model_8.compile(optimizer="adam", loss='mse',metrics=['mse'])
from keras.models import load_model
model_8.load_weights('.\train_weights_183l_frgb.h5')
st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
@st.cache(suppress_st_warning=True)
def main():

  text_holder5.title(" Solar Wind Prediction ")
  html_temp = """
    <div style="background-color:#522a02 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Solar Wind Prediction </h2>
    </div>
    """ 
  st.markdown(html_temp,unsafe_allow_html=True)
def import_and_predict(img,model):
  np_img = np.array(img)
  np_img.resize(512,512,1)
  np_test=np_img.reshape(1,512,512,1)
  prediction=model_8.predict(np_test)
  pred='{0:.{1}f}'.format(prediction[0][0], 2)
  return float(pred)

def import_and_predict2(image_data,model):
  img= Image.open(file)
  np_img = np.array(img)
  np_img.resize(512,512,1)
  np_test=np_img.reshape(1,512,512,1)
  prediction=model_8.predict(np_test)
  pred='{0:.{1}f}'.format(prediction[0][0], 2)
  return float(pred)

def get_latest_value(URL):
  page = urllib.request.urlopen(URL   )
  soup= BeautifulSoup(page , "html.parser")
  text = soup.get_text().split("\n")
  text.reverse()
  text = text[1:]
  print(text)
  for line in text:
    tmp = float(line.split()[8])
    if (tmp) != float(-9999.9):
      return (tmp)

image_holder=st.empty()
text_holder=st.empty()
image_holder2=st.empty()
text_holder2=st.empty()
text_holder3=st.empty()
image_holder3=st.empty()
text_holder4=st.empty()
text_holder5=st.empty()


while(1):
  text_holder5.title(" Solar Wind Prediction ")
 

  file=st.file_uploader("Please upload an image",type=["jpg","png"])
  if file is None:
    text_holder3.text("Please Upload an image")    
  else:
    image=Image.open(file)
    image_holder3.image(image,use_column_width=True) 
    predictions =import_and_predict2(image,model_8)
    text_holder4.success('The solar wind speed for the image is {}'.format(predictions))
  
  URL = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193.jpg"
  URL1 = "https://services.swpc.noaa.gov/text/ace-swepam.txt"
  r = requests.get(URL)
  soup = BeautifulSoup(r.text ,"html.parser")
  im = Image.open(requests.get(URL, stream=True).raw)
  image_holder.image(im,use_column_width=True) 
  predictions = import_and_predict(im,model_8)
  text_holder.success('The solar wind speed for the image is {}'.format(predictions))
  text_holder2.markdown('**The plot for solar wind speed:-**.')
  result = get_latest_value(URL1)
  data={'Actual':result,'Predicted':predictions}
  labels=list(data.keys())
  values=list(data.values())
  fig = plt.figure(figsize=(7,5))
  plt.bar(labels,values,color='green',width=0.4)
  plt.title("Latest | Actual vs Predicted")
  plt.savefig('foo.jpg', bbox_inches='tight',dpi=100,transparent=False)
  img2="/content/foo.jpg"
  image_holder2.image(img2,use_column_width=True)
  time.sleep(900)

  
if __name__=='__main__':
    main()
