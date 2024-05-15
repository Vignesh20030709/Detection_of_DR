import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import PIL
print(os.getcwd())
train_dir = os.getcwd()+'\\train'
print(train_dir)
train_zero_dir = os.getcwd()+'\\train\\zero'
print(train_zero_dir)
train_one_dir = os.getcwd()+'\\train\\one'
train_two_dir = os.getcwd()+'\\train\\two'
train_three_dir = os.getcwd()+'\\train\\three'
train_four_dir  = os.getcwd()+'\\train\\four'

for file in os.listdir(train_zero_dir):
    print(file)
    
count=0
for file in os.listdir(train_one_dir):
    print(file)
    count+=1
for file in os.listdir(train_two_dir):
    print(file)
    count+=1
for file in os.listdir(train_three_dir):
    print(file)
    count+=1
for file in os.listdir(train_four_dir):
    print(file)
    count+=1
    
print(count)

# Let's apply data augmentation on this one folder:
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   )
