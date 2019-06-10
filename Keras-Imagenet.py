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
        result = decode_predictions(preds, top=1)[0]
        print('Predicted:', result[0][1],' ',round((result[0][2])*100),'%',' Time(mS): ',round((end_t-start_t)*1000,2))
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows

exit()









