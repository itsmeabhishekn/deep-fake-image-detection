import cv2
import os

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt
import cv2
from moviepy.editor import * 

parent_dir = r'/home/goku/code/test_pro'

for i in os.listdir(parent_dir):
    print(i)
    item,ext =os.path.splitext(i);
    path = os.path.join(parent_dir, item)
    print(path)
    os.mkdir(path)
    print("Directory '% s' created" % path)

    
    fname = path+'.mp4'
    video = VideoFileClip((fname))
    
    capture = cv2.VideoCapture(fname)

    frameNr = 0

    
    
    while (True):
        
        success, frame = capture.read()
    
        if success:

        # Read the input image
            #img = cv2.imread(frame)

            # Convert into grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Load the cascade
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)


            # Draw rectangle around the faces and crop the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                faces = frame[y:y + h, x:x + w]

                # cv2.imshow("face",faces)
                #print("breaks after this")
                cv2.imwrite('face.jpg', faces)
                
            # Display the output
                cv2.waitKey()
        
                cv2.imwrite(f'{path}/frame_{frameNr}.jpg', frame)
                print(frameNr,"th frame is successfully saved")
        
        else:
            
            break
        
        frameNr = frameNr+1
    
    capture.release()
