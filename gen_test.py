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
    # for step in range(start_frame, start_frame + 10):
    #     bpy.context.scene.frame_set(step)
    take_action(p2_link, start_frame + 20, (dx2, dy2, dz))
    toggle_animation(p2_link, start_frame + 20, False)
    # for i in range(start_frame + 10, frame_offset + 101):
    #     bpy.context.scene.frame_set(i)


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
    # One-hot encoded action labels
    a_enc = []
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

    # If we want to perturb the rope
    perturb = 1
    if perturb: 
        pert2 = random.sample(range(len(rope)), 1)[0]
        random_perturb(pert2, 30, rope, 0)
    a_a = np.array([[15, 2, 2],
                 [15, 1, 2],
                 [15, 1, 2], [15, 0, 2]])
    for t in range(T):
        print('Experiment Step: ', t)
        print('Current offset: ', frame_offset)
        st = []
        for r in rope:
            st_loc = r.matrix_world.to_translation()
            st.append(np.array(st_loc)[:2])
        st = np.array(st)
        # Move the end link
        # keyf = random.sample(range(3, 20), 1)[0]
        keyf = 10
        # Record the random action
        at = np.array([np.random.uniform(0.5, 3) * random.choice((-1, 1)), np.random.uniform(0.5, 3) * random.choice((-1, 1)), np.random.uniform(0.5, 2)])
        # at = a_a[t]
        # Encode the action into one-hot representation using histogram. 
        # Note that the action space is coarsely discretized into arrays separated by 0.1
        at_enc = np.array([np.histogram(at[0], bins = np.linspace(-3, 3, 61))[0],
                           np.histogram(at[1], bins = np.linspace(-3, 3, 61))[0],
                           np.histogram(at[2], bins = np.linspace(0.5, 2, 16))[0]])
        # Take the action step in sim
        if perturb: 
            take_action(rope[-1], frame_offset + 110, (at[0], at[1], at[2]))
        else:
            take_action(rope[-1], frame_offset + 10, (at[0], at[1], at[2]))

        print("Action taken: ", at)
        if not Vis:
            # Then wait for another 100 frames for the rope to settle
            for i in range(frame_offset, frame_offset + 100):
                bpy.context.scene.frame_set(i)
                if i == 50 and perturb:
                    save_render_path = os.path.join(os.getcwd(), 'inv_model_15k_multistep')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'gt_perturb_exp_%d.jpg'%(num))
                    bpy.context.scene.camera.location = (0, 0, 60)
                    bpy.ops.render.render(write_still = True)
                if i % 10 == 0:
                    save_render_path = os.path.join(os.getcwd(), 'inv_model_15k_multistep/video/gt')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'expgt_%d_frame_%03d.jpg'%(num, i))
                    bpy.context.scene.camera.location = (0, 0, 60)
                    bpy.ops.render.render(write_still = True)
        print('\n')
        # blenderenv.add_camera_light()
        s.append(st)
        a.append(at)
        a_enc.append(at_enc)
        frame_offset += 100
        # Delete all keyframes to make a new knot and reset the frame counter
        # bpy.context.scene.frame_set(0)
        # for ac in bpy.data.actions:
        #     bpy.data.actions.remove(ac) 
        save_render_path = os.path.join(os.getcwd(), 'inv_model_15k_multistep')
        bpy.context.scene.render.filepath = os.path.join(save_render_path, 'gt_exp_%d_%d.jpg'%(num, t))
        bpy.context.scene.camera.location = (0, 0, 60)
        bpy.ops.render.render(write_still = True)
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
    np.save(os.path.join(save, 'multistep_demo_actions_encoded.npy'), a_enc)