Real Time Face Recognition and Exam Proctoring using
Deep learning.

The aim of this project is to develop a deep learning model
that can accurately detect and recognize human faces in an
image and can monitor whether the person in front of camera
is involved in any malpractice during the exam or not. Also to
provide a smooth interface between admin and the
user/student.

By developing a deep learning model which can automate this
process and provide users and administration with more
accurate handling of the exams held. It will not only automate
this but also there will be no chance of malpractice and
candidates appearing for exams will get equal opportunity.

Requirements:-
1. Python: 3.10.10
2. Flask: 2.1.2
3. Visual Studio code

The methodology behind exam proctoring system using
deep learning involves the following steps:

1.Face detection: Haar Cascade algorithm is used to detect
human faces via the webcam mounted on students' devices.

2.Data collection: Real time data is collected and saved to
server PC, for every user i.e. student 30 images in .jpeg
format are captured by the device webcam.

3.Data preprocessing: Data collected from the directory is
resized according to the CNN model. Data is then split into
training and testing data as 70% and 20% respectively.
Converts the data into a NumPy Array and scales it between 0
to 1 and one hot encoding is implemented.
4.Training of model: The CNN model is trained on the
processed data. The model is compiled using Adam optimizer
and categorical cross entropy.

5.Real time face recognition: The trained model identifies the
user sitting in front of the camera.

6.Real time face orientation recognition: Media pipe library is
used to detect the face orientation.
