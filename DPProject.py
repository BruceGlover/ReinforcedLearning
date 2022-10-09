#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tensorflow.keras.utils  import  array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest')

import os 
print(os.getcwd()) ##Previous Directory
os.chdir(r"D:\fresho") ##Change with your current working directory
print(os.getcwd())  ##Current Working Directory

for path in os.listdir():
    img = load_img(f"{path}")
    x = img_to_array(img)    # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=".", save_prefix='img', save_format='jpeg'):
        i += 1
        if i > 10:     ## creates 10 image form 1 image 
            break  


# In[4]:


from tensorflow.keras.utils  import  array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest')

import os 
print(os.getcwd()) ##Previous Directory
os.chdir(r"D:\overipo") ##Change with your current working directory
print(os.getcwd())  ##Current Working Directory

for path in os.listdir():
    img = load_img(f"{path}")
    x = img_to_array(img)    # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=".", save_prefix='img', save_format='jpeg'):
        i += 1
        if i > 10:     ## creates 10 image form 1 image 
            break 


# In[6]:


from tensorflow.keras.utils  import  array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest')

import os 
print(os.getcwd()) ##Previous Directory
os.chdir(r"D:\underipeo") ##Change with your current working directory
print(os.getcwd())  ##Current Working Directory

for path in os.listdir():
    img = load_img(f"{path}")
    x = img_to_array(img)    # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=".", save_prefix='img', save_format='jpeg'):
        i += 1
        if i > 10:     ## creates 10 image form 1 image 
            break 


# In[7]:


from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model 
from keras.applications.vgg16 import VGG16 
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np


# In[9]:


### Defining Image size
IMAGE_SIZE = [224, 224]
### Loading model
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
### Freezing layers
for layer in vgg.layers:  
  layer.trainable = False
### adding a 3 node final layer for predicion
x = Flatten()(vgg.output)
prediction = Dense(3, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
### Generating Summary
model.summary()


# In[10]:


model.compile( loss='categorical_crossentropy',  
               optimizer='adam',  
               metrics=['accuracy'])


# In[17]:


train_datagen = ImageDataGenerator(rescale = 1./255,                          
                                    shear_range = 0.2,
                                   zoom_range = 0.2,
                                    horizontal_flip= True)
training_set = train_datagen.flow_from_directory('D:\\underipeo',
                                           target_size = (224, 224),
                                             batch_size = 16,                             
                                        class_mode = 'categorical')


# In[18]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('D:\\Test',
                                         target_size = (224, 224),
                                         batch_size=16,
                                         class_mode = 'categorical')


# In[52]:


r = model.fit_generator(training_set,  validation_data=test_set,  epochs=3,steps_per_epoch=len(training_set),validation_steps=len(test_set))


# In[23]:


model.save("ripeness.h5")


# In[41]:



from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np


# In[54]:



np.set_printoptions(suppress=True)
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Replace this with the path to your image
image = Image.open(r"D:\\trial\\download.jpg")
image.show()
    #resizing the image to be at least 224x224 
    
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    #turn the image into a numpy array
image_array = np.asarray(image)
    
    # Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
data[0] = normalized_image_array
    
        
    # Load the model
model = tensorflow.keras.models.load_model('ripeness.h5')
    
    # run the inference
prediction = model.predict(data)
print(prediction)
    # max_val = np.amax(prediction)*100
    # max_val = "%.2f" % max_val
if np.argmax(prediction)==0:
    print("Unripe")
elif np.argmax(prediction)==1:
    print("Over ripe")
else :
    print("Ripe")




# In[ ]:




