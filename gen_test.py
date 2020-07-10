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

'''Usage: blender -P gen_test.py -- -exp EXP_NUMBER -T SIM_HORIZON -render RENDER_BOOLEAN'''
# def take_action(obj, frame, action_vec, animate=True):
#     # Keyframes a displacement for obj given by action_vec at given frame
#     curr_frame = bpy.context.scene.frame_current
#     dx,dy,dz = action_vec
#     if animate != obj.rigid_body.kinematic:
#         # We are "picking up" a dropped object, so we need its updated location
#         obj.location = obj.matrix_world.translation
#         obj.rotation_euler = obj.matrix_world.to_euler()
#         obj.keyframe_insert(data_path="location", frame=curr_frame)
#         obj.keyframe_insert(data_path="rotation_euler", frame=curr_frame)
#     toggle_animation(obj, curr_frame, animate)
#     obj.location += Vector((dx,dy,dz))
#     obj.keyframe_insert(data_path="location", frame=frame)

def take_action(held_link, at, keyf, settlef):
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + keyf)
    held_link.location += Vector((at[0], at[1], at[2]))
    held_link.keyframe_insert(data_path="location")
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + settlef)
    held_link.keyframe_insert(data_path="location")

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
    parser.add_argument('-T', '--hrzn', dest='hrzn', type=int)
    parser.add_argument('-render', '--render', dest='render', type=int)
    args = parser.parse_known_args(argv)[0]
    num = args.exp_num

    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    
    clear_scene()
    rope = make_rope_v3(params)
    make_table(params)
    bpy.context.scene.gravity *= 10
    add_camera_light()

    # Simulation horizon T
    T = args.hrzn
    # If we want to render images
    render = args.render
    # Visualiztion flag, if true, will render from the frame the action is applied. If False, will only render the final frame.
    Vis = 0

    # Demo states
    s = []
    # Actions (labels)
    a = []
    # One-hot encoded action labels
    a_enc = []

    held_link = rope[-1]
    held_link.rigid_body.kinematic = True
    
    frame_end = 250 * 30
    
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end

    for r in rope:
        r.rigid_body.mass *= 10
    rope[0].rigid_body.mass *= 5

    frame_offset = 0

    bpy.context.scene.frame_set(1)
    for ac in bpy.data.actions:
        bpy.data.actions.remove(ac)

    held_link.keyframe_insert(data_path="location")
    held_link.keyframe_insert(data_path="rotation_euler")
    bpy.context.scene.rigidbody_world.enabled = True
    bpy.context.scene.rigidbody_world.point_cache.frame_start = 1

    # If we want to perturb the rope
    perturb = 0
    if perturb: 
        pert2 = random.sample(range(len(rope)), 1)[0]
        random_perturb(pert2, 30, rope, 0)
    # a_a = np.array([[0, 2, 2],
    #              [0, 2, 1],
    #              [0, 0, -2]])
    for t in range(T):
        print('Experiment Step: ', t)
        print('Current offset: ', frame_offset)
        st = []
        for r in rope:
            st_loc = r.matrix_world.to_translation()
            st.append(np.array(st_loc))
        st = np.array(st)
        # Move the end link
        # keyf = random.sample(range(3, 20), 1)[0]
        keyf = 10
        # Record the random action
        dz = np.random.uniform(-rope[-1].matrix_world.translation[2], 1)
        at = np.array([np.random.uniform(0.5, 2) * random.choice((-1, 1)), np.random.uniform(0.5, 2) * random.choice((-1, 1)), dz])
        # at = a_a[t]
        # Encode the action into one-hot representation using histogram. 
        # Note that the action space is coarsely discretized into arrays separated by 0.1
        at_enc = np.array([np.histogram(at[0], bins = np.linspace(-3, 3, 61))[0],
                           np.histogram(at[1], bins = np.linspace(-3, 3, 61))[0],
                           np.histogram(at[2], bins = np.linspace(0.5, 2, 16))[0]])
        # Take the action step in sim
        if perturb: 
            take_action(held_link, (at[0], at[1], at[2]), 110, 40)
        else:
            take_action(held_link, (at[0], at[1], at[2]), 10, 40)

        print("Action taken: ", at)
        if not Vis:
            # Then wait for another 100 frames for the rope to settle
            for i in range(frame_offset + 1, frame_offset + 51):
                bpy.context.scene.frame_set(i)
                if i == 50 and perturb and render:
                    save_render_path = os.path.join(os.getcwd(), 'inv_model_50k_multistep')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'gt_perturb_exp_%d.jpg'%(num))
                    bpy.context.scene.camera.location = (0, 0, 60)
                    bpy.ops.render.render(write_still = True)
                if i % 10 == 0 and render:
                    save_render_path = os.path.join(os.getcwd(), 'fwd_model_6k_mpc/video/gt')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'expgt_%d_frame_%03d.jpg'%(num, i))
                    bpy.context.scene.camera.location = (5, 0, 60)
                    bpy.ops.render.render(write_still = True)
        print('\n')
        # blenderenv.add_camera_light()
        s.append(st)
        a.append(at)
        a_enc.append(at_enc)
        frame_offset += 50
        # Delete all keyframes to make a new knot and reset the frame counter
        # bpy.context.scene.frame_set(0)
        # for ac in bpy.data.actions:
        #     bpy.data.actions.remove(ac) 
        if render:
            save_render_path = os.path.join(os.getcwd(), 'fwd_model_6k_mpc')
            bpy.context.scene.render.filepath = os.path.join(save_render_path, 'gt_exp_%d_%d.jpg'%(num, t))
            bpy.context.scene.camera.location = (5, 0, 60)
            bpy.ops.render.render(write_still = True)
    st = []
    # Record terminal state
    for r in rope:
        st_loc = r.matrix_world.to_translation()
        st.append(np.array(st_loc))
    st = np.array(st)
    s.append(st)
    endf = 1000
    for frame_no in range(1, endf+1):
        bpy.context.scene.frame_set(frame_no)

    # Save the npy files
    if not os.path.exists("./states_actions/test"):
        os.makedirs('./states_actions/test')
    save = os.path.join(os.getcwd(), 'states_actions/test')
    np.save(os.path.join(save, 'multistep_demo_states_%d.npy'%(num)), s)
    np.save(os.path.join(save, 'multistep_demo_actions_%d.npy'%(num)), a)
    np.save(os.path.join(save, 'multistep_demo_actions_encoded_%d.npy'%(num)), a_enc)