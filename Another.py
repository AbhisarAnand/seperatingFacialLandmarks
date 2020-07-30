import glob
import shutil
import cv2
import numpy



cascPath = "utils/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)



font = cv2.FONT_HERSHEY_SIMPLEX


files = []
ImagesPath = "/*.jpg"
for file in glob.glob(ImagesPath):
    files.append(str(file))

goodImagesPath = "/"
k=0
i=0
while i < len(files):
    # Capture frame-by-frame
    frame = cv2.imread(files[k])
    fframe = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
if faces.isempty():
    shutil.move(files[k], badImagesPath)
else:
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.putText(frame, 'Face', (x, y), font, 2, (255, 0, 0), 5)
        ROI = frame[y:y + h, x:x + w] 
        cv2.imwrite(goodImagesPath+files[k], ROI)
        
        
    i+=1
    k+=1
