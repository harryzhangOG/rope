import numpy as np
import argparse
import os
import cv2
from math import pi

def record_webcam():
    pic_name = 'base_img_eval_knock_new.png'
    d2r = pi / 180
    # Expert policy: midpoint config
    base_mid_config = np.array([  -70.76,  -77.7 , 16.68, -152.95, -118.07, -185. ])*d2r # 66.83 ])
    if not os.path.exists("./physical_exp_datagen"):
        os.makedirs('./physical_exp_datagen')

    cap = cv2.VideoCapture(0)
    num = 0
    pics = []
    while num < 10:
        ret, frame = cap.read()
        vis = frame.copy()
        vis = cv2.resize(vis, (512, 512))
        pics.append(vis)
        num += 1
    cap.release()
    cv2.destroyAllWindows()
    vis = np.zeros_like(pics[0], np.uint8)
    for p in pics:
        vis = cv2.addWeighted(vis, 1, p, 0.1, 0)
    #vis = vis / 10
    cv2.imwrite(os.path.join('./physical_exp_datagen', pic_name), vis)


if __name__ == "__main__":
    record_webcam()
