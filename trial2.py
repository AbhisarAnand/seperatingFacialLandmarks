# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

from os import listdir

import cv2
import dlib
import imutils
# import the necessary packages
import numpy
import time
from imutils import face_utils

path = '/home/abhisar/PycharmProjects/seperatingFacialLandmarks/utils/shape_predictor_68_face_landmarks (1).dat'
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

vs = cv2.VideoCapture("/dev/video0")
fileStream = False
time.sleep(1.0)

files = [] = listdir("idenprof/training/")


# loop over frames from the video stream
while True:

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    _, frame = vs.read()
    frame = imutils.resize(frame, width=450)
    fframe = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEyeHull = cv2.boundingRect(leftEye)
        x, y, w, h = leftEyeHull
        rightEyeHull = cv2.boundingRect(rightEye)
        x1, y1, w1, h1 = rightEyeHull
        mouthHull = cv2.boundingRect(mouth)
        x2, y2, w2, h2 = mouthHull
        a = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
        b = cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), -1)
        c = cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 255, 255), -1)
        white_color = numpy.array([255, 255, 255])
        white_mask = cv2.inRange(frame, white_color, white_color)
        white = cv2.bitwise_and(frame, frame, mask=white_mask)
        finalMask = cv2.copyTo(fframe, white)
        frame = finalMask

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
