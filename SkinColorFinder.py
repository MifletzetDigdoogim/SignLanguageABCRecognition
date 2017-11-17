import cv2
import numpy as np

##########CONSTANTS##########
sqr_side_length = 50
bound_extra = 5
####################

camera = cv2.VideoCapture(0)

skin_rects = []
while True:
    (_, frame) = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    # Added 1 to the padding to make sure border isn't captured in the crop
    cv2.rectangle(frame, (width//2 - sqr_side_length//2 - 1, height//2 - sqr_side_length//2 - 1), (width//2 + sqr_side_length//2 + 1, height//2 + sqr_side_length//2 + 1), (255, 255, 0))
    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord("t"):
        skin_rect = frame[height//2 - sqr_side_length//2 : height//2 + sqr_side_length//2, width//2 - sqr_side_length//2 : width//2 + sqr_side_length//2]
        skin_rects.append(skin_rect)
        print("Snap!")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Closing...")
        break

# cv2.imshow("Skin", skin_rects[0])
h, s, v = [], [], []
for skin_rect in skin_rects:
    for y in range(sqr_side_length):
        for x in range(sqr_side_length):
            # print(str(skin_rect[x, y]))
            h.append(skin_rect[x, y][0])
            s.append(skin_rect[x, y][1])
            v.append(skin_rect[x, y][2])

h.sort()
s.sort()
v.sort()
lower_bound = (h[0] - bound_extra, s[0] - bound_extra, v[0] - bound_extra)
upper_bound = (h[len(h) - 1] + bound_extra, s[len(s) - 1] + bound_extra, v[len(v) - 1] + bound_extra)



print("lower_skin = np.array(" + str(lower_bound) + ")")
print("upper_skin = np.array(" + str(upper_bound) + ")")

cv2.waitKey(0)

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()