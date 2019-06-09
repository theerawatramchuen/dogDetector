# Original Code
# https://keras.io/applications/#classify-imagenet-classes-with-resnet50

import time
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

# Part 3 Prediction Image from video
import numpy as np
from keras.preprocessing import image as image_utils
import numpy as np
import cv2
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    start_t = time.time()
    ret, frame = cap.read()
    
    if ret == True:
        cv2.imshow('frame',frame)
        cv2.imwrite('temp.jpg',frame)
        img_path = 'temp.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        end_t = time.time()
        print('Predicted:', decode_predictions(preds, top=1)[0],' Time: ',round((end_t-start_t),2))
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows

exit()











start_t = time.time()
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
end_t = time.time()
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=1)[0],' Time: ',round((end_t-start_t),2))
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]


start_t = time.time()
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
end_t = time.time()
print('Predicted:', decode_predictions(preds, top=1)[0],' Time: ',round((end_t-start_t),2))


start_t = time.time()
img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
end_t = time.time()
print('Predicted:', decode_predictions(preds, top=1)[0],' Time: ',round((end_t-start_t),2))

