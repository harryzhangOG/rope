import numpy as np
import random
import argparse
import os
import cv2
from ur5_whip_motion import *
from math import pi

def record_webcam(seq_no, dirc):
    pic_name = 'whip_state_%04d.png'%(seq_no)
    pic_name_end = 'whip_state_end_%04d.png'%(seq_no)
    action_name = 'whip_action_%04d.npy'%(seq_no)
    d2r = pi / 180
    # Expert policy: midpoint config
    base_mid_config = np.array([  -75.76,  -62.7 , 22.68, -152.95, -118.07, -185. ])*d2r # 66.83 ])
    #base_mid_config = np.load(os.path.join('repeatability', 'whip_action_0002.npy'))
    #base_mid_config = np.load(os.path.join('redo_knock', 'whip_action_0042.npy'))
    #base_mid_config = np.load('demo/demo_knock.npy')
    #base_mid_config = np.load('demo/demo_vault.npy')
    base_x = base_mid_config[0]/d2r
    base_y = base_mid_config[1]/d2r
    base_z = base_mid_config[2]/d2r
    if not os.path.exists(os.path.join(os.getcwd(), dirc)):
        os.makedirs(os.path.join(os.getcwd(), dirc))

    save_dir = os.path.join(os.getcwd(), dirc)

    success = False
    while not success:
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
        # Generate rope whipping trajectories
        dat_main_datagen(base_mid_config)
        # Execute the trajectory on UR5 by calling this Cpp program

        #os.system("") TODO: INSERT UR5_GO COMMAND HERE. Filename: ur5_whip_datagen.dat. prob need to cd into that dir first 
        os.chdir('../ur5_go')
        #os.system('./build/Debug/ur5_go -h 172.22.22.3 -r 0.1 -t 0.032 ur5_taut_s.dat')
        #os.system('./build/Debug/ur5_go -h 172.22.22.3 -r 0.1 -t 0.032 ur5_taut_reset_s.dat')
        os.system('./build/Debug/ur5_go -h 172.22.22.3 -r 1.0 -t 0.032 ur5_whip_datagen.dat')
        os.chdir('../rope')

        yesChoice = ['yes', 'y']
        noChoice = ['no', 'n']
        # Wait for human annotator to judge if the exec was successful
        inp = input("Success? (y/N) ").lower()

        if inp in yesChoice:
        #if True:
            success = True
            # TODO: read base image and save the difference
#            base_image = cv2.imread('./physical_exp_datagen/base_image.png')
#            vis = cv2.absdiff(vis, base_image)
            pics_end = []
            num = 0
            while num < 10:
                ret, frame = cap.read()
                vis = frame.copy()
                vis = cv2.resize(vis, (512, 512))
                pics_end.append(vis)
                num += 1
            cap.release()
            cv2.destroyAllWindows()
            vis = np.zeros_like(pics[0], np.uint8)
            vis_end = np.zeros_like(pics_end[0], np.uint8)
            for p in pics:
                vis = cv2.addWeighted(vis, 1, p, 0.1, 0)
            for p in pics_end:
                vis_end = cv2.addWeighted(vis_end, 1, p, 0.1, 0)
            cv2.imwrite(os.path.join(save_dir, pic_name), vis)
            cv2.imwrite(os.path.join(save_dir, pic_name_end), vis_end)
            np.save(os.path.join(save_dir, action_name), base_mid_config)
#            dat_main_datagen_reset(base_mid_config)
            # TODO: RESET THE ROBOT FOR NEXT TRIAL
            os.chdir('../ur5_go')
            os.system('./build/Debug/ur5_go -h 172.22.22.3 -r 1.0 -t 0.032 ur5_reset_datagen.dat')
            os.chdir('../rope')
            return
        elif inp in noChoice:
#            dat_main_datagen_reset(base_mid_config)
            # If failed, reset the robot to starting config
            #os.system("") TODO: INSERT UR5_GO RESET HERE. Filename: ur5_reset_datagen.dat
            cap.release()
            cv2.destroyAllWindows()
            os.chdir('../ur5_go')
            os.system('./build/Debug/ur5_go -h 172.22.22.3 -r 1.0 -t 0.032 ur5_reset_datagen.dat')
            os.chdir('../rope')
            # Random search for suitable residual values
            base_mid_config = np.array([base_x + np.random.uniform(4, 10)*random.choice((-1, 1)), 
                                        base_y + np.random.uniform(4, 10)*random.choice((-1, 1)), 
                                        base_z + np.random.uniform(4, 8),
                                        -152.95,
                                        -118.07,
                                        -185])*d2r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=int, help='index of current trial')
    parser.add_argument('--dirc', type=str, help='datagen directory', default='physical_exp_datagen')

    args = parser.parse_args()
    seq_no = args.seq
    dirc = args.dirc
    record_webcam(seq_no, dirc)

