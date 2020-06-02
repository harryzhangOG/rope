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
from rigidbody_rope import *

'''Usage: blender -P eval_rope.py'''
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

def set_st(pert2, dx2, dy2, start_frame, rope, end_frame):
    p2_link = rope[pert2]
    dz = 0
    print("Perturbation 1: ", pert2, dx2, dy2)
    for step in range(start_frame, start_frame + 10):
        bpy.context.scene.frame_set(step)
    take_action(p2_link, start_frame + 20, (dx2, dy2, dz))
    toggle_animation(p2_link, start_frame + 20, False)
    for i in range(start_frame + 10, 101):
        bpy.context.scene.frame_set(i)

def move_rope_end(end, target, key_f):
    """
    Move the end link of the rope.
    END: the link you are moving. A blender object.
    TARGET: the target location you are moving to. A (x, y, z) tuple.
    """
    end.rigid_body.kinematic = True
    bpy.context.scene.frame_set(30)
    end.location = end.matrix_world.to_translation()
    end.keyframe_insert(data_path="location", frame=30)
    end.keyframe_insert(data_path="rotation_euler", frame=30)
    key_frames = [[key_f, 0, target]]
    for fno, rot, loc in key_frames:
        bpy.context.scene.frame_current = fno
        end.location += Vector(loc)
        end.keyframe_insert(data_path="location", frame=fno)
    for i in range(50):
        bpy.context.scene.frame_set(i)
if __name__ == "__main__":
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
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
    # Perturbation taken to make st
    pert = [10, -0.6862172912046929, 1.3118705817974519]
    pert2 = pert[0]
    dx2 = pert[1]
    dy2 = pert[2]
    for i in range(50):
        bpy.context.scene.frame_set(i)
        if i == 30:
            take_action(rope[-1], i, (0, 0, 0))
            toggle_animation(rope[-1], i, False) 
            take_action(rope[pert2], i, (0, 0, 0))
            toggle_animation(rope[pert2], i, False) 
    # Set st in sim
    set_st(pert2, dx2, dy2, 30, rope, frame_end)
    # Take action
    gt = [ 8.     ,    -0.86478555, -1.56295504]
    # take_action(rope[-1], 100 + gt[0], (gt[1], gt[2], 0))
    pred = [ 9,  -1.3281376, -1.7353866]
    take_action(rope[-1], 100 + pred[0], (pred[1], pred[2], 0))
    # for i in range(100, 200):
    #     bpy.context.scene.frame_set(i)
    


