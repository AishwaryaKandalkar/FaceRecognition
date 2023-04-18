# from My_dl import predicted_classes
import cv2
import tensorflow
import numpy as np
import keras
from numpy import asarray

our_model= keras.models.load_model('model.h5')

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
names = ['None', 'A', 'B', 'C', 'D', 'E'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    # ret, img =cam.read()
    # img = cv2.flip(img, 1) # Flip vertically
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # # cvtColor() method is used to convert an image from one color space to another.
    # faces = faceCascade.detectMultiScale( 
    #     gray,
    #     scaleFactor = 1.2,
    #     minNeighbors = 5,
    #     minSize = (int(minW), int(minH)),
    #    )
    # gray=gray.resize(gray,(64,64))
    # image_pixels = asarray(gray)
    # image_pixels = np.expand_dims(image_pixels, axis = 0)
    # # image_pixels=image_pixels.resize(image_pixels,(64,64))
    # image_pixels = image_pixels.astype('float64')
    # image_pixels /= 255
    # for(x,y,w,h) in faces:
    #     cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    #     # id = predicted_classes
    #     my_img = gray[y:y+h,x:x+w]
    #     id,confidence = our_model.predict(image_pixels)
    
    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # cvtColor() method is used to convert an image from one color space to another.
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        new_gray = gray[y:y+h,x:x+w]
        # img_data = cv2.re
        img_data = cv2.resize(gray, (64, 64))
        confidence = our_model.predict(img_data.reshape(1,64,64,1))
        id = np.argmax(confidence[0])
        print(id)
        name = names[id]
        
        
        cv2.putText(
                    img, 
                    str(name), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
       
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()