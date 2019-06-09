# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense

# Initialise the number of classes
num_classes = 2
 
# Build the model
classifier = Sequential()
classifier.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
classifier.add(Dense(num_classes, activation='softmax'))
 
# Say not to train first layer (ResNet) model. It is already trained
classifier.layers[0].trainable = False
 
# Compiling the CNN
classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()

# Loading model weight
start = time.time()
classifier.load_weights('cat_dog_rasnet50.h5')

# Part 3 Prediction Image from video
import numpy as np
from keras.preprocessing import image as image_utils
import time
import cv2
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    start = time.time()
    ret, frame = cap.read()
    
    if ret == True:
        cv2.imshow('frame',frame)
        cv2.imwrite('temp.jpg',frame)
        test_image = image_utils.load_img('temp.jpg', target_size = (64, 64))
        test_image = image_utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict_on_batch(test_image)
        print ('Dog ',round(result[0][1]*100.0),'%')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows
