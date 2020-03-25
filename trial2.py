import glob
import shutil

import cv2
import dlib
import numpy
from imutils import face_utils
from pip._vendor.distlib.compat import raw_input

path = '/home/abhisar/PycharmProjects/seperatingFacialLandmarks/utils/shape_predictor_68_face_landmarks (1).dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


files = []
ImagesPath = "/home/abhisar/Desktop/idenprof/test/Asleep/*.jpg"
for file in glob.glob(ImagesPath):
    files.append(str(file))

badImagesPath = "/home/abhisar/Desktop/idenprof/badPictures"
higherColor = [10, 10, 10]

k = 0
i = 0
while i < len(files):
    frame = cv2.imread(files[k])
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    print(files[k])
    anotherFrame = frame.copy()
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
        finalMask = cv2.copyTo(anotherFrame, white)
        frame = finalMask
        cv2.imshow("Frame", frame)
        cv2.imwrite(files[k], frame)
        print("saving")
    pixels = numpy.float32(frame.reshape(-1, 3))
    n_colors = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = numpy.unique(labels, return_counts=True)
    dominant = palette[numpy.argmax(counts)]
    if dominant[0] > higherColor[0]:
        print("Delete the file: ", files[k])
        shutil.move(files[k], badImagesPath)
        cv2.imshow("Deleted Frames", frame)
    k = k + 1
    i += 1
