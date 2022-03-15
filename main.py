#references : https://youtu.be/pp61TbhJOTg
#ieee paper link : https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=brain+tumor+detection&oq=brain+#d=gs_qabs&u=%23p%3DVe4XM0A9CzEJ
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import load_model

image_dir='C:/Users/kotta/PycharmProjects/braintumordetection/dataset/'
INPUT_SIZE=64
dataset=[]
label=[]

no_tumor_images=os.listdir(image_dir+ 'no/')
yes_tumor_images=os.listdir(image_dir+ 'yes/')


for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.cvtColor(cv2.imread(image_dir+'no/'+image_name), cv2.COLOR_BGR2GRAY)
        image=cv2.resize(image, (INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.cvtColor(cv2.imread(image_dir+'yes/'+image_name), cv2.COLOR_BGR2GRAY)
        image=cv2.resize(image, (INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)
label=np.array(label)
#x train and x test is images
#y train and y test is digit representation of images

x_train, x_test, y_train ,y_test = train_test_split(dataset, label, test_size=0.2, random_state=0 )
x_train, x_val, y_train ,y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0 )
y_train= tf.keras.utils.to_categorical(y_train, num_classes=2)#it convert vector class to binary class matrix
y_val= tf.keras.utils.to_categorical(y_val, num_classes=2)
y_test= tf.keras.utils.to_categorical(y_test, num_classes=2)


#Model Building

model=tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32,(3,3),activation='ReLU', input_shape=(INPUT_SIZE, INPUT_SIZE, 1)),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Conv2D(32,(3,3), kernel_initializer='he_uniform'),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Conv2D(32,(3,3), kernel_initializer='he_uniform'),
tf.keras.layers.MaxPooling2D((2,2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(2, activation='sigmoid'),
])

model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#checkpoint = keras.callbacks.ModelCheckpoint("Best_1.h5",save_best_only=True) #confusion matrix
history=model.fit(x_train,
          y_train,
          batch_size=16,
          verbose=1,
          epochs=10,
          validation_data=(x_val,y_val),
          shuffle=False)

model.save('BrainTumor10EpochsCategorical.h5')

y = model.predict(x_test)
model.evaluate(x_test, y_test)
print(confusion_matrix(np.argmax(y_test, 1), np.argmax(y, 1)))

pd.DataFrame(history.history).plot(figsize = (15,8))
plt.gca().set_ylim(0,1)
plt.show()
#model1 = model.load_model("Best_1.h5")
#accuracy = model1.evaluate(x_test,y_test)
#accuracy[1]

#pred = (model1.predict(x_test) > 0.5).astype("int32")


#sns.heatmap(confusion_matrix(y_test,pred), annot = True, cmap = "rainbow")
#plt.title("Confusion Matrix")
#plt.show()
# RATIO_LIST = []
# for set in (x_train, x_test, x_val):
#     for img in set:
#         RATIO_LIST.append(img.shape[1] / img.shape[0])
#
# plt.hist(RATIO_LIST)
# plt.title('Distribution of Image Ratios')
# plt.xlabel('Ratio Value')
# plt.ylabel('Count')
# plt.show()

#values = ["No Tumor Detected","Tumor Detected"]
#y_t = np.array(y_test)
#pred = np.array(pred).reshape(1,-1)

#i = 0
#for data in x_test:
#    print("Predicted :- {} with {}% accuracy".format(values[pred[0][i]],round(accuracy[1]*100,2)))
#    print("Actual :- {}".format(values[y_t[i]]))
 #   plt.imshow(data, cmap="gray")
  #  plt.show()
   # i = i+1
