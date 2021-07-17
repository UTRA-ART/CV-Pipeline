'''
Example implementation and use of Frame_handler class 
'''


import time
import os

import cv2
import matplotlib.pyplot as plt
from src.frame_handler import Frame_Handler

times = []
handler = Frame_Handler()

for i in [1, 4, 5]:
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(fileDir, '.\data\\') + str(i) + ".mp4"

    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break
        
        handler.update_from_camera(img)

        start = time.time()
        handler.process_frame()
        times += [time.time() - start]

        cv2.imshow("Frame Detections", handler.drawn_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('n'):
            break

print("Frame rate", round(1/(sum(times[5:])/len(times[5:]))),"Hz")

plt.plot(times[5:])
plt.show()

cap.release()
cv2.destroyAllWindows()