import face_recognition
import numpy as np
import os
import cv2

images = []
classNames = []
myList = os.listdir('../potos')
print(myList)
for cl in myList:
    cur_img = cv2.imread(f'potos/{cl}')
    images.append(cur_img)
    if (os.path.splitext(cl)[1] == '.jpg' or os.path.splitext(cl)[1] == '.jpeg' or os.path.splitext(cl)[1] == '.png'):
        classNames.append(os.path.splitext(cl)[0])

#print(images)

def findEncodings(images):
    encodedList = []
    for img in images:
        #img=np.full((100,80,3),12,dtype=np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        encoding = face_recognition.face_encodings(img)[0]
        encodedList.append(encoding)
    return encodedList


encodedListKnown = findEncodings(images)
#print(encodedListKnown)


cap = cv2.VideoCapture(0)

while True:
    succes,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_RGB2BGR)
    faceCurLocations = face_recognition.face_locations(imgS)
    faceCurEncodings = face_recognition.face_encodings(imgS,faceCurLocations)
    for faceEnco,faceLoc in zip(faceCurEncodings,faceCurLocations):
        matches = face_recognition.compare_faces(encodedListKnown,faceEnco)
        #print(matches)
        distances = face_recognition.face_distance(encodedListKnown,faceEnco)
        print(distances)
        matchIndex = np.argmin(distances)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            x1,x2,y1,y2 = faceLoc
            x1,x2,y1,y2 = x1*4,x2*4,y1*4,y2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            cv2.rectangle(img,(x1,y1-245),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow("Webcam ",img)
    cv2.waitKey(1)

            

