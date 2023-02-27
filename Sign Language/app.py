#Import all necessary libraries
import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt
import time
import mediapipe as mp

#Defining the Holistic models and drawing utilities variables
mp_holistic = mp.solutions.holistic        #Holistic model; Make the detections
mp_drawing = mp.solutions.drawing_utils     #Drawing utilities; Draw the detections

#Functions for reuse of Holistic models and drawing utilities variables for detections
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Will pass the Image and the Holistic Model in Blue_Green_Red to Red_Green_Blue; Colour Conversion using cv2.cvtColor
    image.flags.writeable = False                  #Sets image writeable to False
    results = model.process(image)                 #Making the Prediction
    image.flags.writeable = True                   #Sets image writeable to True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #Will pass the Image and the Holistic Model in Red_Green_Blue TO Blue_Green_Red; Colour Conversion
    return image, results                          #Returns the image and result into the loop

#Function for reuse of draw_landmarks; 
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)        #Will draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)        #Will draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)   #Will draw left_hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  #Will draw right_hand connections

def draw_styled_landmarks(image, results):
    #Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                              ) #BGR to RGB Landmark and Connection
    
    #Draw pose connections     
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                              ) #BGR to RGB Landmark and Connection
     
    #Draw left_hand connections        
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                              ) #BGR to RGB Landmark and Connection
      
     #Draw right_hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,110,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                              ) #BGR to RGB Landmark and Connection  

#Access MediaPipe Holistic Model
cap  = cv2.VideoCapture(0)                 #Accessing web cam using variable cap

#Set MediaPipe Model
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic: #Able to access the holistic model by creating detention confidence and tracking it by 0.5

    while cap.isOpened():                      #Initiate loop in web cam
    
        #Read Feed
        ret, frame = cap.read()                #Read frame from prescribed time, If stacked together will look like a video capture. This will return Value and Frame
        
        #Make Detection
        image, results =mediapipe_detection(frame, holistic)
        print (results)

        #Draw Landmarks
        draw_landmarks(image, results)

        #Show to Screen
        cv2.imshow('OpenCV Feed', frame)       #Will show frame and have them placed in OpenCV Feed

        #Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'): #Wait for 'q' key to be pressed to break from the loop

            break

    cap.release()  #Realese the Web Cam
    cv2.destroyAllWindows()  #Destroy all Windows

len(results.pose_landmarks.landmark) #Their are face_landmarks(), count(), index, left_hand_landmarks(), mro(), pose_landmarks(), right_hand_landmarks()

frame

results

draw_landmarks(frame, results)

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #Convert to real original color of the frame 