
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set the image size
IMAGE_SIZE = [224, 224]

# Load the InceptionV3 model pre-trained on ImageNet data
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze all layers in the base model
for layer in inception.layers:
    layer.trainable = False

# Flatten the output layer to 1 dimension
x = Flatten()(inception.output)

# Add a fully connected layer with softmax activation for 2 classes
prediction = Dense(5, activation='softmax')(x)

# Create the model
model = Model(inputs=inception.input, outputs=prediction)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Show model summary
model.summary()

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_set = train_datagen.flow_from_directory('D:/Code_Diabetic_Retinopathy_Inception/train',
                                              target_size=(224, 224),
                                              batch_size=32,
                                              class_mode='categorical')
test_set = test_datagen.flow_from_directory('D:/Code_Diabetic_Retinopathy_Inception/test',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')

# Train the model
r = model.fit(train_set, validation_data=test_set, epochs=5, steps_per_epoch=len(train_set), validation_steps=len(test_set))

# Plot loss
if 'loss' in r.history:
    plt.plot(r.history['loss'], label='train loss')
if 'val_loss' in r.history:
    plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Plot accuracy
if 'accuracy' in r.history:
    plt.plot(r.history['accuracy'], label='train acc')
if 'val_accuracy' in r.history:
    plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

# Save the model
model.save('inception_model.h5')
