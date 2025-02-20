##pip install tensorflow pandas matplotlib opencv-python


# upload dataset to drive and mount it
from google.colab import drive
drive.mount('/content/drive')


img_path = "/content/drive/MyDrive/final/"
act_label = ['accident' , 'theft' , 'fire' , 'normal']
import pandas as pd
import os
import tensorflow as tf
import numpy as np

img_list = []
label_list = []
for label in act_label:
    for img_file in os.listdir(img_path+label):
        img_list.append(img_path+label+'/'+img_file)
        label_list.append(label)

df = pd.DataFrame({'img':img_list, 'label':label_list})


df_labels = {
    'accident' : 0,
    'theft' : 1,
    'fire' : 2,
    'normal' : 3,
    }
df['encode_label'] = df['label'].map(df_labels)
df.head()
import cv2


X = []

for img in df['img']:
    ano = img
    img = cv2.imread(str(img))
    # img = augment_function(img)
    if (img is not None):
        img = cv2.resize(img, (96, 96))
        img = img/255
        X.append(img)
    else:
        print(ano)



y = df['encode_label']
from sklearn.model_selection import train_test_split
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val)




from keras.applications.vgg16 import VGG16

base_model = VGG16(input_shape=(96,96,3), include_top=False, weights='imagenet')

base_model.summary()
for layer in base_model.layers:
    layer.trainable = False
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
model = Sequential()
model.add(Input(shape=(96,96,3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(act_label), activation='softmax'))
model.summary()
model.compile(
  optimizer="adam",
  loss='sparse_categorical_crossentropy',
  metrics=['acc'])


X_train = tf.convert_to_tensor(np.array(X_train))
X_val = tf.convert_to_tensor(np.array(X_val))
y_train = tf.convert_to_tensor(y_train.values)
y_val = tf.convert_to_tensor(y_val.values)



history = model.fit(tf.stack(X_train), tf.stack(y_train), epochs=5, validation_data=(X_val, y_val))
print(history)







# Predict the class of the image
def predict_image(image_path, model, label_map):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load the image. Please check the file path.")
        return None, None
    img = cv2.resize(img, (96, 96))      
    img = img / 255.0                     
    img = np.expand_dims(img, axis=0)     
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)   
    predicted_label = label_map[predicted_class_index]   
    confidence = np.max(predictions)   
    return predicted_label, confidence




label_map = {0: 'accident', 1: 'theft', 2: 'fire', 3: 'normal'}
user_image_path = input('Enter image path  :  ')
predicted_label, confidence = predict_image(user_image_path, model, label_map)
if predicted_label is not None:
    print(f"Predicted class: {predicted_label}, Confidence: {confidence:.2f}")







# print accuracy
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
accuracy = accuracy*100
print("Accuracy:  {:.2f}".format(accuracy))


