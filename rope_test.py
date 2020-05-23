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
import datetime

'''Usage: blender -P rope_test.py'''
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
    print('Experiment Number: ', num)

    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)

    # Source state
    st = []
    # Target state
    stp1 = []
    # Actions (labels)
    at = []
    
    # Make a new Blender Env
    blenderenv = BlenderEnv(params)
    # Make a new rope
    blenderenv.clear_scene()
    blenderenv.make_table(params)
    frame_end = 300
    rope = blenderenv.make_rope("capsule_12_8_1_2.stl")
    bpy.context.scene.frame_current = 0
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end

    rope[0].rigid_body.mass *= 5
    rope[-1].rigid_body.mass *= 2
    for r in rope:
        st_loc = r.matrix_world.to_translation()
        st.append(np.array(st_loc)[:2])
    st = np.array(st)
    # Randomly perturb the rope
    idx = random.randint(0, len(rope))
    idx_target = [rope[idx].location[0], rope[idx].location[1] + random.uniform(-2.5, 2.5), rope[idx].location[2]]
    move_rope_end(rope[idx], idx_target, 10)
    
    # Move the end link
    target = (random.uniform(-13, -10), random.uniform(-3, 3), 0)
    keyf = random.randint(2, 15)
    # Record the action
    at = np.array([keyf, target[0] - rope[-1].location[0], target[1] - rope[-1].location[1]])
    move_rope_end(rope[-1], target, keyf)
    print("Action taken: ", at)
    # All links' locations:
    for r in rope:
        stp1_loc = r.matrix_world.to_translation()
        print("New state location: ", stp1_loc)
        stp1.append(np.array(stp1_loc)[:2])
    stp1 = np.array(stp1)
    blenderenv.add_camera_light()
    # Save the npy files
    if not os.path.exists("./states_actions"):
        os.makedirs('./states_actions')
    save = os.path.join(os.getcwd(), 'states_actions')
    np.save(os.path.join(save, 's_t_' + str(num) + '.npy'), st)
    np.save(os.path.join(save, 's_tp1_' + str(num) + '.npy'), stp1)
    np.save(os.path.join(save, 'a_t_' + str(num) + '.npy'), at)

        # Quit Blender
        # bpy.ops.wm.quit_blender()

