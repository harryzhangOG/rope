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

'''Usage: blender -P rope_test.py'''
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

def random_perturb(pert2, start_frame, rope, end_frame):
    p2_link = rope[pert2]
    dz = 0
    dx2 = np.random.uniform(0.5, 2) * random.choice((-1, 1))
    dy2 = np.random.uniform(0.8, 2) * random.choice((-1, 1))
    print("Perturbation 1: ", pert2, dx2, dy2)
    for step in range(start_frame, start_frame + 10):
        bpy.context.scene.frame_set(step)
    take_action(p2_link, start_frame + 20, (dx2, dy2, dz))
    toggle_animation(p2_link, start_frame + 20, False)
    for i in range(start_frame + 10, 101):
        bpy.context.scene.frame_set(i)
# def random_perturb(start_frame, rope, end_frame):
#     pert1, pert2 = 0, 0
#     while abs(pert1 - pert2) <= 5:
#         pert1, pert2 = random.sample(range(len(rope)), 2)
#     p1_link = rope[pert1]
#     p2_link = rope[pert2]
#     dx1 = np.random.uniform(0.5, 1) * random.choice((-1, 1))
#     dy1 = np.random.uniform(0.5, 1) * random.choice((-1, 1))
#     dz = 0
#     dx2 = np.random.uniform(0.5, 1) * random.choice((-1, 1))
#     dy2 = np.random.uniform(0.5, 1) * random.choice((-1, 1))
#     print("Perturbation 1: ", pert1, dx1, dy1)
#     print("Perturbation 2: ", pert2, dx2, dy2)
#     for step in range(start_frame, start_frame + 10):
#         bpy.context.scene.frame_set(step)
    
#     take_action(p1_link, start_frame, (dx1, dy1, dz))
#     take_action(p2_link, start_frame + 10, (dx2, dy2, dz))
#     toggle_animation(p1_link, start_frame + 20, False)
#     toggle_animation(p2_link, start_frame + 20, False)
#     for i in range(start_frame + 10, start_frame+30):
#         bpy.context.scene.frame_set(i)

def move_rope_end(end, target, key_f):
    """
    Move the end link of the rope.
    END: the link you are moving. A blender object.
    TARGET: the target location you are moving to. A (x, y, z) tuple.
    """
    end.rigid_body.kinematic = True
    end.keyframe_insert(data_path="location", frame=1)
    end.keyframe_insert(data_path="rotation_euler", frame=1)
    key_frames = [[key_f, 0, target]]
    for fno, rot, loc in key_frames:
        bpy.context.scene.frame_current = fno
        end.location = loc
        end.rotation_euler = (rot*pi/180, 90*pi/180, 0)
        end.keyframe_insert(data_path="location", frame=fno)
        end.keyframe_insert(data_path="rotation_euler", frame=fno)
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
    T = 5000
    # Source state
    s = []
    # Target state
    sp1 = []
    # Actions (labels)
    a = []
    # # Make a new Blender Env
    # blenderenv = BlenderEnv(params)
    # # Make a new rope
    # clear_scene()
    # blenderenv.make_table(params)
    # # rope = blenderenv.make_rope("capsule_12_8_1_2.stl")
    # rope = make_capsule_rope(params)
    # blenderenv.rig_rope(params)
    clear_scene()
    rope = make_capsule_rope(params)
    rope[0].rigid_body.mass *= 5
    rope[-1].rigid_body.mass *= 2
    rig_rope(params)
    add_camera_light()
    frame_end = 300
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end
    make_table(params)
    for t in range(T):
        print('Experiment Number: ', t)
        st = []
        stp1 = []
        # Randomly perturb the rope
        # idx = random.randint(0, len(rope) - 1)
        # idx_target = [rope[idx].matrix_world.to_translation()[0], rope[idx].matrix_world.to_translation()[1] + random.uniform(-2.5, 2.5), rope[idx].matrix_world.to_translation()[2]]
        # print("Perturbation: ", idx, idx_target)
        # move_rope_end(rope[idx], idx_target, 10)
        pert2 = random.sample(range(len(rope)), 1)[0]
        for i in range(50):
            bpy.context.scene.frame_set(i)
            if i == 30:
               take_action(rope[-1], i, (0, 0, 0))
               toggle_animation(rope[-1], i, False) 
               take_action(rope[pert2], i, (0, 0, 0))
               toggle_animation(rope[pert2], i, False) 
        random_perturb(pert2, 30, rope, frame_end)
        # Wait for the rope to settle in the scene
        
        # FIRST, WAIT 100 FRAMES, this was done in random perturb.
        # Record S_t at frame 100
        for r in rope:
            st_loc = r.matrix_world.to_translation()
            st.append(np.array(st_loc)[:2])
        st = np.array(st)
        
        # Move the end link
        # target = (random.uniform(-13, -10), random.uniform(-3.5, 3.5), 0)
        # keyf = random.randint(5, 20)
        keyf = random.sample(range(3, 20), 1)[0]
        # Record the random action
        at = np.array([keyf, np.random.uniform(0.5, 3) * random.choice((-1, 1)), np.random.uniform(0.5, 3) * random.choice((-1, 1))])
        # Take the action step in sim
        take_action(rope[-1], 100 + at[0], (at[1], at[2], 0))
        print("Action taken: ", at)
        # Then wait for another 100 frames for the rope to settle
        for i in range(100, 200):
            bpy.context.scene.frame_set(i)
        # Record all links' locations as s_t+1 at frame 200:
        for r in rope:
            stp1_loc = r.matrix_world.to_translation()
            # print("New state location: ", stp1_loc)
            stp1.append(np.array(stp1_loc)[:2])
        stp1 = np.array(stp1)
        # Checking if the output contains nan
        # print("Wrong output number: ", np.count_nonzero(np.isnan(stp1)))
        print('\n')
        # blenderenv.add_camera_light()
        s.append(st)
        sp1.append(stp1)
        a.append(at)
        # Delete all keyframes to make a new knot and reset the frame counter
        bpy.context.scene.frame_set(0)
        for ac in bpy.data.actions:
            bpy.data.actions.remove(ac) 
    

    # Save the npy files
    if not os.path.exists("./states_actions"):
        os.makedirs('./states_actions')
    save = os.path.join(os.getcwd(), 'states_actions')
    np.save(os.path.join(save, 's.npy'), s)
    np.save(os.path.join(save, 'sp1.npy'), sp1)
    np.save(os.path.join(save, 'a.npy'), a)
