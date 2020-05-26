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

'''Usage: blender -P eval_rope.py'''
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
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    # Make a new Blender Env
    blenderenv = BlenderEnv(params)
    # Make a new rope
    blenderenv.clear_scene()
    blenderenv.make_table(params)
    rope = blenderenv.make_rope("capsule_12_8_1_2.stl")
    
    frame_end = 300
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end

    rope[0].rigid_body.mass *= 5
    rope[-1].rigid_body.mass *= 2
    # Load the test source state s_t
    s_t = np.load(os.path.join('states_actions', "s_test.npy"))[2]
    # Load the test target state s_tp1
    s_tp1 = np.load(os.path.join('states_actions', "sp1_test.npy"))[0]
    # for i, r in enumerate(rope, 0):
    #     move_rope_end(r, (s_t[i][0], s_t[i][1], 0), 10)
    # Configure the rope according to the loaded source states
    pert_idx = 77
    pert_target = [-9.925000190734863, 1.2632782587929094, 0.0]
    move_rope_end(rope[pert_idx], pert_target, 10)
    for i in range(20):
        bpy.context.scene.frame_set(i)
    # Move the rope according to the predicted action
    a_t_pred = [9, 1.8388214,  0.24663843]
    move_rope_end(rope[-1], (a_t_pred[1] + rope[-1].matrix_world.to_translation()[0], 
                             a_t_pred[2] + rope[-1].matrix_world.to_translation()[1], 0), 
                             a_t_pred[0] + 30)


