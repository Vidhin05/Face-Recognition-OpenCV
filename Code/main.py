import cv2
import sys
from mail import sendEmail
from flask import Flask, render_template, Response
from camera import VideoCamera
import time
import numpy as np
import threading
import os


email_update_interval = 60 # sends an email only once in this time interval
object_classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") # an opencv classifier

Subjects = ["","Deep Thanki", "Rutvik Shah" , "Unknown"]

time.sleep(10)

def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5 , minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):

    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for dir_name in dirs:

        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:

            if image_name.startswith("."):
                continue

            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path) 
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels

cv2.destroyAllWindows()
faces, labels = prepare_training_data("training-data")
face_recognizer = cv2.face.createLBPHFaceRecognizer(2,16,7,7)
face_recognizer.train(faces, np.array(labels))

print("Testing over!!");

video_camera = VideoCamera(flip=True) 

# App Globals (do not edit)
app = Flask(__name__)
last_epoch = 0

def check_for_objects():
	global last_epoch
	while True:
		frame = video_camera.get_frame()
		original = frame.copy()
		gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
		face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
		faces = face_cascade.detectMultiScale(gray, 1.1 , 5 , )


		for img in faces:
			x,y,w,h = img
			label= face_recognizer.predict(gray[x:x+w,y:y+h])
			cv2.rectangle(original, (x,y) , (x+w,y+h) , (0,255,0) , 1)
			name = Subjects[label]
			cv2.putText(original, name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
		
		cv2.imshow("face_recognizer",original)
		ret, jpeg = cv2.imencode('.jpg', original)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			
		if len(faces) > 0  and (time.time() - last_epoch) > email_update_interval:
			last_epoch = time.time()
			print "Sending email..."
			sendEmail(jpeg.tobytes())
			print "done!"

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame_np()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    t = threading.Thread(target=check_for_objects, args=())
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', debug=False)
