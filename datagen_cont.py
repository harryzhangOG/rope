import bpy
import os
import json
import time
import sys
import bpy, bpy_extras
from math import *
from mathutils import *
import random
import numpy as np
import random
from random import sample
import bmesh
sys.path.append(os.getcwd())
from blender_rope import *
from rigidbody_rope import *
import argparse

def take_action(held_link, at, keyf, settlef):
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + keyf)
    held_link.location += Vector((at[0], at[1], at[2]))
    held_link.keyframe_insert(data_path="location")
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + settlef)
    held_link.keyframe_insert(data_path="location")

if "__main__" == __name__:
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-num', '--num_iterations', dest='num_iterations', type=int)

    args = parser.parse_known_args(argv)[0]
    # Number of episodes
    N = args.num_iterations

    # Source state
    s = []
    # Target state
    sp1 = []
    # Actions
    a = []

    clear_scene()
    add_camera_light()
    rope = make_rope_v3(params)
    make_table(params)
    # bpy.context.scene.gravity *= 10
    frame_end = 250 * 30
    
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end

    held_link = rope[-1]
    held_link.rigid_body.kinematic = True
    init_loc = Vector(held_link.location)

    for r in rope:
        r.rigid_body.mass *= 10
    r.rigid_body.mass *= 5
    for seq_no in range(N):
        print('Experiment Number: ', seq_no)
        # remove all keyframes
        bpy.context.scene.frame_set(1)
        for ac in bpy.data.actions:
            bpy.data.actions.remove(ac)

        held_link.keyframe_insert(data_path="location")
        held_link.keyframe_insert(data_path="rotation_euler")
        bpy.context.scene.rigidbody_world.enabled = True
        bpy.context.scene.rigidbody_world.point_cache.frame_start = 1

        # set up key frames
        for action_no in range(20):
            keyf = 5
            settlef = 1
            # dz needs some special handling
            dz = np.random.uniform(-held_link.location[2], 1)
            # Record the random action
            at = np.array([np.random.uniform(0.1, 1.25) * random.choice((-1, 1)), np.random.uniform(0.1, 1.25) * random.choice((-1, 1)), dz])
            print("Action taken: ", at)
            a.append(at)

            take_action(held_link, at, keyf, settlef)

        # play back animation
        endf = bpy.context.scene.frame_current
        for frame_no in range(1, endf+1):
            bpy.context.scene.frame_set(frame_no)
            record_state = (frame_no % 6 == 1)
            if record_state and frame_no == 1:
                st = []
                for r in rope:
                    st_loc = r.matrix_world.to_translation()
                    st_vel = [0, 0, 0]
                    st.append(np.concatenate([np.array(st_loc), np.array(st_vel)]))
                s.append(st)
                # Debugging : render the images
                # bpy.context.scene.render.filepath = '/Users/harryzhang/Desktop/st_%d.jpg'%(frame_no)
                # bpy.context.scene.camera.location = (0, 0, 60)
                # bpy.ops.render.render(write_still = True)
            elif record_state and frame_no != 1 and frame_no != endf:
                st = []
                stp1 = []
                for i, r in enumerate(rope, 0):
                    st_loc = r.matrix_world.to_translation()
                    # Calculate velocity
                    st_vel = (np.array(st_loc) - s[-1][i][:3])/5
                    st.append(np.concatenate([np.array(st_loc), st_vel]))
                    stp1.append(np.concatenate([np.array(st_loc), st_vel]))
                s.append(st)
                sp1.append(stp1)
                # Debugging : render the images
                # bpy.context.scene.render.filepath = '/Users/harryzhang/Desktop/st_%d.jpg'%(frame_no)
                # bpy.context.scene.camera.location = (0, 0, 60)
                # bpy.ops.render.render(write_still = True)
            elif frame_no == endf:
                stp1 = []
                for i, r in enumerate(rope, 0):
                    stp1_loc = r.matrix_world.to_translation()
                    # Calculate velocity
                    stp1_vel = (np.array(stp1_loc) - sp1[-1][i][:3])/5 
                    stp1.append(np.concatenate([np.array(stp1_loc), np.array(stp1_vel)]))
                sp1.append(stp1)
                # Debugging : render the images
                # bpy.context.scene.render.filepath = '/Users/harryzhang/Desktop/st_%d.jpg'%(frame_no)
                # bpy.context.scene.camera.location = (0, 0, 60)
                # bpy.ops.render.render(write_still = True)
    # Save the npy files
    if not os.path.exists("./states_actions_cont"):
        os.makedirs('./states_actions_cont')
    save = os.path.join(os.getcwd(), 'states_actions_cont')
    np.save(os.path.join(save, 'a.npy'), a)
    np.save(os.path.join(save, 's.npy'), s)
    np.save(os.path.join(save, 'sp1.npy'), sp1)