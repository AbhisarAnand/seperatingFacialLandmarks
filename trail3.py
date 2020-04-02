import glob

import cv2


files = []
ImagesPath = "/home/abhisar/Desktop/idenprof/test/asleep/*.jpg"
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

    frame = cv2.resize(frame, (320, 240))

    print(frame.shape)

    cv2.imshow("Frame", frame)
    cv2.imwrite(files[k], frame)
    print("saving")
    k = k + 1
    i += 1
