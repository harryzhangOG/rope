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

def take_at(obj, frame, action_vec, animate=True):
    # Keyframes a displacement for obj given by action_vec at given frame
    curr_frame = bpy.context.scene.frame_current
    dx,dy,dz = action_vec
    if animate != obj.rigid_body.kinematic:
        # We are "picking up" a dropped object, so we need its updated location
        obj.location = (obj.matrix_world.translation[0], obj.matrix_world.translation[1], 0)
        obj.keyframe_insert(data_path="location", frame=curr_frame)
    toggle_animation(obj, curr_frame, animate)
    obj.location += Vector((dx,dy,dz))
    obj.keyframe_insert(data_path="location", frame=frame)

def toggle_animation(obj, frame, animate):
    # Sets the obj to be animable or non-animable at particular frame
    obj.rigid_body.kinematic = animate
    obj.keyframe_insert(data_path="rigid_body.kinematic", frame=frame)

def random_perturb(start_frame, rope, end_frame):
    pert1, pert2 = random.sample(range(len(rope)), 2)
    p1_link = rope[pert1]
    p2_link = rope[pert2]
    dx1 = np.random.uniform(0.5, 1) * random.choice((-1, 1))
    dy1 = np.random.uniform(0.8, 2) * random.choice((-1, 1))
    dz = 0
    dx2 = np.random.uniform(0.5, 1) * random.choice((-1, 1))
    dy2 = np.random.uniform(0.8, 2) * random.choice((-1, 1))
    
    take_action(rope[45], start_frame + 10, (0.6, 0.7, 0))
    # toggle_animation(p1_link, start_frame + 10, False)
    # toggle_animation(p2_link, start_frame + 10, False)
    take_action(rope[12], start_frame + 10, (0.5, -0.3, dz))
    for i in range(50):
        bpy.context.scene.frame_set(i)

    # pert1, pert2 = random.sample(range(len(rope)), 2)
    # p1_link = rope[pert1]
    # p2_link = rope[pert2]
    # dx1 = np.random.uniform(0.5, 1) * random.choice((-1, 1))
    # dy1 = np.random.uniform(0.5, 1) * random.choice((-1, 1))
    # dz = 0
    # dx2 = np.random.uniform(0.5, 1) * random.choice((-1, 1))
    # dy2 = np.random.uniform(0.5, 1) * random.choice((-1, 1))
    # print("Perturbation 1: ", pert1, dx1, dy1)
    # print("Perturbation 2: ", pert2, dx2, dy2)
    
    # take_action(p1_link, start_frame + 10, (dx1, dy1, dz))
    # # toggle_animation(p1_link, start_frame + 10, False)
    # # toggle_animation(p2_link, start_frame + 10, False)
    # take_action(p2_link, start_frame + 10, (dx2, dy2, dz))
    # for i in range(50):
    #     bpy.context.scene.frame_set(i)

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
    # Make a new Blender Env
    blenderenv = BlenderEnv(params)
    # Make a new rope
    blenderenv.clear_scene()
    blenderenv.make_table(params)
    rope = blenderenv.make_rope("capsule_12_8_1_2.stl")
    
    frame_end = 100
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end

    rope[0].rigid_body.mass *= 5
    rope[-1].rigid_body.mass *= 2
    # Load the test source state s_t
    s_t = np.load(os.path.join('states_actions', "s_test.npy"))[0]
    # Load the test target state s_tp1
    s_tp1 = np.load(os.path.join('states_actions', "sp1_test.npy"))[0]
    # Configure the rope according to the loaded source states
    # The way we restore the original st is by applying the same perturbation.
    pert_idx = 60
    pert_target = [-5.25, 2, 0.0]
    # move_rope_end(rope[pert_idx], pert_target, 10)
    
    random_perturb(10, rope, frame_end)
    bpy.context.scene.frame_set(0)
    rope[-1].location = rope[-1].matrix_world.translation
    rope[-1].keyframe_insert(data_path="location", frame=0)
    # for r in rope:
        # print(r.matrix_world.translation)
    for i in range (21):
        bpy.context.scene.frame_set(i)
        print(rope[-1].matrix_world.translation)

    take_action(rope[-1], 40, (2, -1, 0))
    # IMPORTANT: set s_t to be reproducible by recording all the current states and pass into keyframes.
    # for i, r in enumerate(rope, 0):
    #     if i == pert_idx:
    #         continue
    #     r.keyframe_insert(data_path="location", frame=1)
    #     r.keyframe_insert(data_path="rotation_euler", frame=1)
    #     loc, rot, scale = r.matrix_world.decompose()
    #     r.rigid_body.kinematic = True
    #     r.location = loc
    #     r.rotation_euler = rot.to_euler('XYZ')
    #     r.keyframe_insert(data_path="location", frame=10)
    #     r.keyframe_insert(data_path="rotation_euler", frame=10)
    # # Move the rope according to the predicted action
    # a_t_pred = [10,    1.9861134, -4.1936927]
    # rope[-2].rigid_body.kinematic = True
    # bpy.context.scene.frame_current = 10 + a_t_pred[0]
    # rope[-2].location = (a_t_pred[1] + rope[-2].matrix_world.to_translation()[0], a_t_pred[2] + rope[-2].matrix_world.to_translation()[1], 0)
    # rope[-2].keyframe_insert(data_path="location", frame=a_t_pred[0] + 10)
    # for i in range(10 + a_t_pred[0]):
    #     bpy.context.scene.frame_set(i)
    


