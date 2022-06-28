import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date

path = 'image Attence'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for file in myList:
    currentImage = cv2.imread(f'{path}/{file}')
    images.append(currentImage)
    classNames.append(os.path.splitext(file)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodeImage = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImage)
    return encodeList

def resize(frame,factor=0.5):
    height = int(frame.shape[0]*factor)
    width = int(frame.shape[1]*factor)
    dimension= (width,height)
    return  cv2.resize(frame,dimension)

serialNo = 0
def markAttendance(name):
    with open('record.csv','r+') as f:
        myDataList = f.readlines()
        namelist = []
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist and name!="Unknown Face":
            now = datetime.now()
            dataString  = now.strftime('%H::%M::%S')
            dates = date.today()
            dateToday = dates.strftime('%d-%m-%Y')
            global serialNo
            serialNo = serialNo + 1
            f.writelines(f'\n{name},{serialNo},{dataString},{dateToday}')

encodeListOfKnownFaces = findEncodings(images)
print('Encoding is Done !!')

#cap  = cv2.VideoCapture('video.mp4')
cap  = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    faceInCurrentFrame = face_recognition.face_locations(imgSmall)
    encodeOfCurrentFrame = face_recognition.face_encodings(imgSmall,faceInCurrentFrame)

    for encodeFace,faceLoc in zip(encodeOfCurrentFrame,faceInCurrentFrame):
        matches = face_recognition.compare_faces(encodeListOfKnownFaces,encodeFace)
        faceDis = face_recognition.face_distance(encodeListOfKnownFaces,encodeFace)
        matchIndex = np.argmin(faceDis)
        print(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
        else:
            print('Unknown Face')
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, 'Unknown Face', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance('Unknown Face')

    now = datetime.now()
    dataString = now.strftime('%H::%M::%S')
    cv2.putText(img, "COUNT : " + str(serialNo), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, dataString, (460, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "MANIT-Project", (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)