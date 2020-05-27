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
        end.keyframe_insert(data_path="location", frame=fno)
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
    s_t = np.load(os.path.join('states_actions', "s_test.npy"))[0]
    # Load the test target state s_tp1
    s_tp1 = np.load(os.path.join('states_actions', "sp1_test.npy"))[0]
    # Configure the rope according to the loaded source states
    pert_idx = 60
    pert_target = [-5.25, 2, 0.0]
    move_rope_end(rope[pert_idx], pert_target, 10)
    for i in range(20):
        bpy.context.scene.frame_set(i)
    bpy.context.scene.frame_set(10)
    # IMPORTANT: set s_t to be reproducible.
    for i, r in enumerate(rope, 0):
        if i == pert_idx:
            continue
        r.keyframe_insert(data_path="location", frame=1)
        r.keyframe_insert(data_path="rotation_euler", frame=1)
        loc, rot, scale = r.matrix_world.decompose()
        r.rigid_body.kinematic = True
        r.location = loc
        r.rotation_euler = rot.to_euler('XYZ')
        r.keyframe_insert(data_path="location", frame=10)
        r.keyframe_insert(data_path="rotation_euler", frame=10)
    # Move the rope according to the predicted action
    a_t_pred = [10,    1.9861134, -4.1936927]
    rope[-1].rigid_body.kinematic = True
    bpy.context.scene.frame_current = 10 + a_t_pred[0]
    rope[-1].location = (a_t_pred[1] + rope[-1].matrix_world.to_translation()[0], a_t_pred[2] + rope[-1].matrix_world.to_translation()[1], 0)
    rope[-1].keyframe_insert(data_path="location", frame=a_t_pred[0] + 10)
    for i in range(10 + a_t_pred[0]):
        bpy.context.scene.frame_set(i)


