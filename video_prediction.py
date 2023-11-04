# Impport necessary libraries and modules
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np

import playsound
from threading import Thread

# Created a dictionary of class numbers and the categories
IMG_CATEGORIES = {0: 'Safe Driving', 1: 'Texting - Right', 2: 'Talking on phone - Right', 3: 'Texting - Left',
                  4: 'Talking on phone - Left', 5: 'Adjusting Radio', 6: 'Drinking', 7: 'Reaching Behind',
                  8: 'Hair and Makeup', 9: 'Talking to Passenger'}

# Specifying the size of image, same as used for training
IMG_SIZE = 224

# Load the model for predictions
pred_model = load_model(r'D:/Users/DELL/Desktop/DS340W/Final_DL_Model.hdf5')

cv2Font = cv2.FONT_HERSHEY_SIMPLEX

current_prediction = 'Driving'
alert = False
writer = None


# Function for loading the alarm tone for alert system
def beep_function():
    playsound.playsound(r'D:\Users\DELL\Desktop\DS340W\beep.mp3', True)


# Capturing the video from source
vc = cv2.VideoCapture(r'D:\Users\DELL\Desktop\DS340W\driving_video.mp4')

writer = None
(W, H) = (None, None)

while True:
    (grabbed, frame) = vc.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    output_frame = frame.copy()

    # Processing every frame obtained from the video
    img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype("float16")

    prediction = pred_model.predict(np.expand_dims(new_array, axis=0))

    result = np.argmax(prediction, axis=1)

    current_prediction = IMG_CATEGORIES.get(int(result))

    if result == 0:
        # Safe driving, text displayed in GREEN
        cv2.putText(output_frame, "Driver's activity: " + current_prediction, (50, 50), cv2Font, 0.7, (0, 255, 0), 2)

    else:
        # Unsafe driving, text displayed in RED
        cv2.putText(output_frame, "Driver's activity: " + current_prediction, (50, 50), cv2Font, 0.7, (0, 0, 255), 2)
        thread = Thread(target=beep_function)
        thread.start()

    # For '.mp4' format
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        writer = cv2.VideoWriter(r'D:\Users\DELL\Desktop\DS340W\video_output/output_video.mp4', fourcc, 30, (W, H),
                                 True)

    # Writing all the frames for video
    writer.write(output_frame)

    # Displaying the output on screen
    cv2.imshow("Output", output_frame)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("q"):
        break

writer.release()
vc.release()