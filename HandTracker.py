import cv2
import numpy as np

from NeuralNetwork import NeuralNetwork
from DataExtractor import Extractor

from gtts import gTTS
from pygame import mixer
import os
from threading import Thread
import time

###################Constants###################
lower_skin = np.array((-5, 72, 51))
upper_skin = np.array((30, 234, 232))
rect_padding = 25
###############################################
##########METHODS##########
def fix_size(width, height, image):
    try:
        blank =  np.zeros((height, width, 3), np.uint8)
        x_offset = blank.shape[1]//2 - image.shape[1]//2
        y_offset = blank.shape[0]//2 - image.shape[0]//2
        blank[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
        image = blank
        return image
    except:
        pass
def get_white_points(image):
    points = []
    h, w = image.shape[:2]
    for y in range(h):
        for x in range(w):
            if image[x, y] == 255:
                points.append((y, x))
    return np.array(points)
def say():
    fc = 0
    while True:
        print("----------------------")
        print("----------------------")
        print("----------------------")
        print(text)
        print("----------------------")
        print("----------------------")
        print("----------------------")

        file_path = "C:\\Users\\Shpoozipoo\\Desktop\\AudioFiles\\label"
        # First make sure there are no files with that name already
        # try:
        #     os.remove(file_path)
        # except OSError:
        #     pass
        # Saving it as a .mp3 file

        tts = gTTS(text=text, lang='en')
        tts.save(file_path + str(fc) + ".mp3")


        mixer.init()
        mixer.music.load(file_path+ str(fc) + ".mp3")
        mixer.music.play()
        fc += 1
        time.sleep(5)

def get_last_index(label, path):
    files = os.listdir(path)
    max_index = -1
    for file in files:
        if str(file).split("-")[0] == label:
            if int(str(file).split("-")[1][:-4]) > max_index:
                max_index = int(str(file).split("-")[1][:-4])
    return max_index


# Get the webcam (0)
camera = cv2.VideoCapture(0)

nn = NeuralNetwork()
LABEL = str(input("Enter the Label \n")) # Label has to be a number. Start from zero.
start = False
counter = get_last_index(LABEL, "C:\\Users\\Shpoozipoo\\Desktop\\Hands\\ABC\\") + 1
text = "Starting!"

thread = Thread(target = say)
thread.start()

while True:
    # grab the current frame
    (is_grabbed_successfully, frame) = camera.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
    mask = cv2.erode(mask, None, iterations=0)
    mask = cv2.dilate(mask, None, iterations=0)

    skin_frame = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
    hand_frame = None
    # Find rectangle only if there are contours
    if contours:
        # Find biggest contour
        hand = max(contours, key=cv2.contourArea)
        # make sure contour isn't tiny
        if cv2.contourArea(hand) > 1000:
            # Draw the biggest contour only
            cv2.drawContours(skin_frame, [hand], 0, (255, 255, 255), 1)
            rect = cv2.boundingRect(hand) # Straight rectangle ; Does not account for angled objects.
            x_r, y_r, w_r, h_r = rect
            # Draw the Rectangle
            # cv2.rectangle(skin_frame, (x_r, y_r), (x_r + w_r, y_r + h_r), (255, 255, 255), 5)
            hand_frame = skin_frame[y_r:y_r + h_r, x_r:x_r + w_r]

    try:
        hand_frame.any()
    except Exception as e:
        continue
    height, width = hand_frame.shape[:2]
    hand_frame = fix_size(max(height, width), max(height, width), hand_frame)
    hand_frame = cv2.resize(hand_frame, (100, 100))
    hand_frame = cv2.cvtColor(hand_frame, cv2.COLOR_HSV2BGR)
    hand_frame = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)
    _, hand_frame  = cv2.threshold(hand_frame,1,255,cv2.THRESH_BINARY)
    # wp = get_white_points(hand_frame)
    # hand_frame = cv2.fillConvexPoly(hand_frame, wp, 255)
    cv2.imshow("Hand", hand_frame)
    if start:
        cv2.imwrite("C:\\Users\\Shpoozipoo\\Desktop\\Hands\\ABC\\" + LABEL + "-" + str(counter) + ".png", hand_frame)
        counter += 1
        print("Saved image number ", counter)

    # cv2.imshow("Skin", skin_frame)
    mirror_frame = cv2.flip(frame, 1)
    cv2.imshow("Normal", mirror_frame)

    if cv2.waitKey(1) & 0XFF == ord("q"):
        break
    if cv2.waitKey(1) & 0xFF == ord("s"):
        start = not start
        print("Starting") if start else print("Stopping")
        if not start:
            LABEL = str(input("Enter the label: \n"))
            counter = 0
    # Predict
    e = Extractor("")
    fs = e.extract_features(hand_frame)
    corresponding_names = ["A", "B", "C", "D", "E", "F", "G", "H" , "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    text = corresponding_names[int(nn.predict(fs))]



# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()