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

'''Usage: blender -P rope_test.py'''
def move_rope_end(end, target):
    """
    Move the end link of the rope.
    END: the link you are moving. A blender object.
    TARGET: the target location you are moving to. A (x, y, z) tuple.
    """
    end.rigid_body.kinematic = True
    end.keyframe_insert(data_path="location", frame=1)
    end.keyframe_insert(data_path="rotation_euler", frame=1)
    key_frames = [[0, 0, (-13.225, 0, 0)],
                  [20, 0, target]]
    for fno, rot, loc in key_frames:
        bpy.context.scene.frame_current = fno
        end.location = loc
        end.rotation_euler = (rot*pi/180, 90*pi/180, 0)
        end.keyframe_insert(data_path="location", frame=fno)
        end.keyframe_insert(data_path="rotation_euler", frame=fno)

if __name__ == "__main__":
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    
    # Make a new Blender Env
    blenderenv = BlenderEnv(params)
    # Make a new rope
    blenderenv.clear_scene()
    rope = blenderenv.make_rope("capsule_12_8_1_2.stl")
    blenderenv.make_table(params)
    frame_end = 30
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end

    rope[0].rigid_body.mass *= 5
    rope[-1].rigid_body.mass *= 2
    
    # Move the end link
    target = (-13.225, 3, 0)
    move_rope_end(rope[-1], target)
    blenderenv.add_camera_light()

