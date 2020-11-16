import numpy as np
import argparse
import os
import cv2
import torch
from torchvision import models, transforms
import torch.nn as nn
from train_ur5_whip_physical import DistModel
from ur5_whip_motion import *
from math import pi
from PIL import Image

def segment_online(base_img, raw_img, seq_no):
    diff = cv2.absdiff(base_img, raw_img)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    th = 40
    imask = mask > th
    canvas = np.zeros_like(raw_img, np.uint8)
    canvas[imask] = raw_img[imask]
    canvas[:120, :, :] = 0
    canvas[100:250, :150, :] = 0

    cv2.imwrite('physical_knock_eval/seg_%04d.png'%(seq_no), canvas)

def record_webcam(seq_no):
    pic_name = 'knock_eval_%04d.png'%(seq_no)
    d2r = pi / 180
    # Expert policy: midpoint config
    if not os.path.exists("./physical_knock_eval"):
        os.makedirs('./physical_knock_eval')

    cap = cv2.VideoCapture(0)
    # Take average images across 10 frames
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
    cv2.imwrite(os.path.join('./physical_knock_eval', pic_name), vis)
    base_img = cv2.imread('./physical_exp_datagen/base_img_eval_knock_new.png')

    # Segment the images
    segment_online(base_img, vis, seq_no)
    
    # Generate rope whipping trajectories using neural net generated residual vel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net50 = DistModel(device, 6)
    state_dict = torch.load('resnet_ur5_model_physical_knock_2000.pth', map_location=device)['model_state_dict']
    net50.load_state_dict(state_dict)
    net50.resnet_mean.to(device)

    in_image = Image.open('physical_knock_eval/seg_%04d.png'%(seq_no)).convert("RGB")
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    in_image = preprocess(in_image).unsqueeze(0)

    net50.eval()
    pred_mid_config = net50(in_image.to(device)).sample().cpu().numpy()[0]
    print(pred_mid_config)

    dat_main_datagen(pred_mid_config)
    # Execute the trajectory on UR5 by calling this Cpp program

    os.chdir('../ur5_go')
    os.system('./build/Debug/ur5_go -h 172.22.22.2 -r 1.0 -t 0.032 ur5_whip_datagen.dat')
    os.chdir('../rope')

    yesChoice = ['yes', 'y']
    noChoice = ['no', 'n']
    # Wait for human annotator to judge if the exec was successful
    inp = input("Success? (y/N) ").lower()

    if inp in yesChoice:
        success = True
        # TODO: read base image and save the difference
        # TODO: RESET THE ROBOT FOR NEXT TRIAL
        os.chdir('../ur5_go')
        os.system('./build/Debug/ur5_go -h 172.22.22.2 -r 1.0 -t 0.032 ur5_reset_datagen.dat')
        os.chdir('../rope')
        return
    elif inp in noChoice:
#        dat_main_datagen_reset(base_mid_config)
        # If failed, reset the robot to starting config
        #os.system("") TODO: INSERT UR5_GO RESET HERE. Filename: ur5_reset_datagen.dat
        cap.release()
        cv2.destroyAllWindows()
        os.chdir('../ur5_go')
        os.system('./build/Debug/ur5_go -h 172.22.22.2 -r 1.0 -t 0.032 ur5_reset_datagen.dat')
        os.chdir('../rope')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=int, help='index of current trial')

    args = parser.parse_args()
    seq_no = args.seq
    record_webcam(seq_no)

