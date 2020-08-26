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
from ur5_viz import UR5
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def take_action(held_link, at, keyf, settlef):
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + keyf)
    held_link.location += Vector((at[0], at[1], at[2]))
    held_link.keyframe_insert(data_path="location")
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + settlef)
    held_link.keyframe_insert(data_path="location")

if "__main__" == __name__:
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    add_camera_light()
    rope = make_rope_v3(params)
    make_table(params)
    frame_end = 25 * 30
    
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end

    held_link = rope[-1]
    held_link.rigid_body.kinematic = True

    # If we want to fix the free end
    fix_free_end = True
    rope[0].rigid_body.kinematic = fix_free_end

    for r in rope:
        r.rigid_body.mass *= 10
    r.rigid_body.mass *= 5

    bpy.context.scene.frame_set(1)
    for ac in bpy.data.actions:
        bpy.data.actions.remove(ac)

    keyf = np.random.randint(10, 20)

    ur5 = UR5()
    # 1. Scale the UR5 base and move it
    ur5.base.scale[0] = 5
    ur5.base.scale[1] = 5
    ur5.base.scale[2] = 5
    ur5.base.location[2] = -1
    bpy.context.view_layer.update()
    ur5.keyframe_insert(1)

    # 2a. A keyframe motion of held link to put in gripper (start kf = 1, end kf = 50)
    at = ur5.gripper.right_inner_finger_pad.matrix_world.translation - held_link.matrix_world.translation
    print(at)
    held_link.keyframe_insert(data_path="location")
    held_link.keyframe_insert(data_path="rotation_euler")
    bpy.context.scene.rigidbody_world.enabled = True
    bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
    take_action(held_link, at, keyf, 50-keyf)
    # 2b. Set keyframe to ur5
    ur5.keyframe_insert(51)
    # 3. Add rigid-body constraint to attach held end to ur5.gripper.gripper_base
    bpy.context.view_layer.objects.active = ur5.gripper.right_inner_finger_pad
    bpy.context.view_layer.objects.active = held_link
    bpy.ops.rigidbody.connect(con_type='FIXED')
    # 4. make held link.kinematic=False
    # held_link.rigid_body.kinematic = False
    # 5. ur5.set_config(...)
    # 6. ur5.keyframe_insert(...)



