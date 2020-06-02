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
import datetime

'''Usage: blender -P gen_test.py -- -exp 0'''
def take_action(obj, frame, action_vec, animate=True):
    # Keyframes a displacement for obj given by action_vec at given frame
    curr_frame = bpy.context.scene.frame_current
    dx,dy,dz = action_vec
    if animate != obj.rigid_body.kinematic:
        # We are "picking up" a dropped object, so we need its updated location
        obj.location = obj.matrix_world.translation
        obj.rotation_euler = obj.matrix_world.to_euler()
        obj.keyframe_insert(data_path="location", frame=curr_frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=curr_frame)
    toggle_animation(obj, curr_frame, animate)
    obj.location += Vector((dx,dy,dz))
    obj.keyframe_insert(data_path="location", frame=frame)


def toggle_animation(obj, frame, animate):
    # Sets the obj to be animable or non-animable at particular frame
    obj.rigid_body.kinematic = animate
    obj.keyframe_insert(data_path="rigid_body.kinematic", frame=frame)

def random_perturb(pert2, start_frame, rope, frame_offset):
    p2_link = rope[pert2]
    dz = 0
    dx2 = np.random.uniform(0.5, 2) * random.choice((-1, 1))
    dy2 = np.random.uniform(0.8, 2) * random.choice((-1, 1))
    print("Perturbation 1: ", pert2, dx2, dy2)
    np.save(os.path.join(os.getcwd(), 'states_actions/multistep_pert.npy'), np.array([pert2, dx2, dy2]))
    for step in range(start_frame, start_frame + 10):
        bpy.context.scene.frame_set(step)
    take_action(p2_link, start_frame + 20, (dx2, dy2, dz))
    toggle_animation(p2_link, start_frame + 20, False)
    for i in range(start_frame + 10, frame_offset + 101):
        bpy.context.scene.frame_set(i)


import argparse
if __name__ == "__main__":
    # Add command line args support (omg this is hard)
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--exp_num', dest='exp_num', type=int)
    args = parser.parse_known_args(argv)[0]
    num = args.exp_num

    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    
    # Simulation horizon T
    T = 4
    # Visualiztion flag, if true, will render from the frame the action is applied. If False, will only render the final frame.
    Vis = 0
    # Demo states
    s = []
    # Actions (labels)
    a = []
    clear_scene()
    rope = make_capsule_rope(params)
    rope[0].rigid_body.mass *= 5
    rope[-1].rigid_body.mass *= 2
    rig_rope(params)
    add_camera_light()
    frame_end = 3000
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end
    make_table(params)
    frame_offset = 0
    # Randomly perturb the rope
    pert2 = random.sample(range(len(rope)), 1)[0]
    for i in range(frame_offset, frame_offset + 50):
        bpy.context.scene.frame_set(i)
        if i == frame_offset + 30:
           take_action(rope[-1], i, (0, 0, 0))
           toggle_animation(rope[-1], i, False) 
           take_action(rope[pert2], i, (0, 0, 0))
           toggle_animation(rope[pert2], i, False) 
    random_perturb(pert2, 30, rope, 0)
    for t in range(T):
        print('Experiment Step: ', t)
        print('Current offset: ', frame_offset)
        st = []
        for r in rope:
            st_loc = r.matrix_world.to_translation()
            st.append(np.array(st_loc)[:2])
        st = np.array(st)
        # Move the end link
        keyf = random.sample(range(3, 20), 1)[0]
        # Record the random action
        at = np.array([keyf, np.random.uniform(0.5, 3) * random.choice((-1, 1)), np.random.uniform(0.5, 3) * random.choice((-1, 1))])
        # Take the action step in sim
        take_action(rope[-1], frame_offset + 100 + at[0], (at[1], at[2], 0))
        print("Action taken: ", at)
        if not Vis:
            # Then wait for another 100 frames for the rope to settle
            for i in range(frame_offset + 100, frame_offset + 200):
                bpy.context.scene.frame_set(i)
        print('\n')
        # blenderenv.add_camera_light()
        s.append(st)
        a.append(at)
        frame_offset += 200
        # Delete all keyframes to make a new knot and reset the frame counter
        # bpy.context.scene.frame_set(0)
        for ac in bpy.data.actions:
            bpy.data.actions.remove(ac) 
    st = []
    # Record terminal state
    for r in rope:
        st_loc = r.matrix_world.to_translation()
        st.append(np.array(st_loc)[:2])
    st = np.array(st)
    s.append(st)

    # Save the npy files
    if not os.path.exists("./states_actions"):
        os.makedirs('./states_actions')
    save = os.path.join(os.getcwd(), 'states_actions')
    np.save(os.path.join(save, 'multistep_demo_states.npy'), s)
    np.save(os.path.join(save, 'multistep_demo_actions.npy'), a)