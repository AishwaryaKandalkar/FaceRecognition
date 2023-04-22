from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_detector = cv2.CascadeClassifier('data.json')
face_id = input('\n enter user id: ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")


def gen_frames():
    count = 0
    while(True):
        ret, img = cam.read()
        img = cv2.flip(img, 1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("data/User." + str(face_id) + '.' +  str(count) + ".jpg", gray[y:y+h,x:x+w])
        # cv2.imwrite("data/User." + str(face_id) + '.' +  str(count) + ".jpg", img)
            #cv2.imwrite() method is used to save an image to any storage device
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 10: # Take 30 face sample and stop video
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")

@app.route('/')
def index():
   return render_template('login.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame'),render_template('login.html')

if __name__ == '__main__':
   app.run(debug = True)