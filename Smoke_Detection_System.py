import cv2
import numpy as np
from keras.models import load_model
from playsound import playsound

model = load_model("smoke_detection_model.h5")
#now we are going to create a function to detct fire
def detect_smoke(frame, threshold=0.6): # we can increase the threshold to make less sensitive and low threshold to make high sensitive
    # I have use the  threshold  0.6 which good model and we can say  a good sesitive.


##############


    # threshold =0.6 is working for smoke and face video (but in face video for few seconds he can detect the smoke due to the light in the room)
    #model can detect smoke inreality the smoke will not be there. 
    #it is detected due to the lights effects according to the background.


    ##########
    preprocess_frame = cv2.cvtColor(cv2.resize(frame,(48,48)), cv2.COLOR_BGR2GRAY)
    preprocess_frame = np.expand_dims(preprocess_frame, axis=0)
    preprocess_frame = np.expand_dims(preprocess_frame, axis=-1)
    preprocess_frame = preprocess_frame.astype("float32")/255

    #now we are going to make predictions
    prediction = model.predict(preprocess_frame)
    if prediction[0] [1] >= threshold:
        return True
    else:
        return False
    #now we are going to open the video file

def trigger_alarm():
    playsound(r"D:\Smoke & Fire Detection\alarm_smoke.mp3")  # Play the alarm sound from specified path


cap = cv2.VideoCapture("D:/Smoke & Fire Detection/Dataset/Video/Smoke.mp4") #put the video or image to detect the SMOKE.
# now we are going to check the video file opened successfully or not
if not cap.isOpened():
    print("Error: Could not opened the Video File")
    exit()

    #now we are going to read and process each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # now we are going to detect if there is a fire in the video or not
    if detect_smoke(frame):
        cv2.rectangle(frame, (100,100),(frame.shape[1]-100, frame.shape[0]-100),(0,0,255),2)
        cv2.putText(frame, "Warning: SMOKE IS DETECTED", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        trigger_alarm()
        # now are to display the frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()                    