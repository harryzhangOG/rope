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
import argparse
sys.path.append('/usr/local/lib/python3.6/dist-packages')
from pyvirtualdisplay import Display
# start a fake display
Display().start()

def take_action(held_link, at, keyf, settlef):
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + keyf)
    held_link.location += Vector((at[0], at[1], at[2]))
    held_link.keyframe_insert(data_path="location")
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + settlef)
    held_link.keyframe_insert(data_path="location")

if "__main__" == __name__:
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-num', '--num_iterations', dest='num_iterations', type=int)

    args = parser.parse_known_args(argv)[0]
    # Number of episodes
    N = args.num_iterations

    # Source state
    s = []
    # Target state
    sp1 = []
    # Actions
    a = []

    clear_scene()
    add_camera_light()
    rope = make_rope_v3(params)
    make_table(params)
    # bpy.context.scene.gravity *= 10
    frame_end = 250 * 30
    
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end

    held_link = rope[-1]
    held_link.rigid_body.kinematic = True
    init_loc = Vector(held_link.location)

    for r in rope:
        r.rigid_body.mass *= 10
    r.rigid_body.mass *= 5
    for seq_no in range(N):
        print('Experiment Number: ', seq_no)
        # remove all keyframes
        bpy.context.scene.frame_set(1)
        for ac in bpy.data.actions:
            bpy.data.actions.remove(ac)

        held_link.keyframe_insert(data_path="location")
        held_link.keyframe_insert(data_path="rotation_euler")
        bpy.context.scene.rigidbody_world.enabled = True
        bpy.context.scene.rigidbody_world.point_cache.frame_start = 1

        # set up key frames
        for action_no in range(20):
            keyf = 10
            settlef = 40
            # dz needs some special handling
            dz = np.random.uniform(-held_link.location[2], 2)
            # Record the random action
            at = np.array([np.random.uniform(0.2, 2.5) * random.choice((-1, 1)), np.random.uniform(0.2, 2.5) * random.choice((-1, 1)), dz])
            print("Action taken: ", at)
            a.append(at)

            take_action(held_link, at, keyf, settlef)

        # play back animation
        endf = bpy.context.scene.frame_current
        for frame_no in range(1, endf+1):
            bpy.context.scene.frame_set(frame_no)
            record_state = (frame_no % 50 == 1)
            if record_state and frame_no == 1:
                if not os.path.exists("./mpc_policy_sa/images"):
                    os.makedirs('./mpc_policy_sa/images/s')
                # Get the scene
                scene = bpy.context.scene
                # Set render resolution
                scene.render.resolution_x = 256
                scene.render.resolution_y = 256
                scene.render.resolution_percentage = 100
                save_render_path = os.path.join(os.getcwd(), 'mpc_policy_sa/images/s')
                bpy.context.scene.render.filepath = os.path.join(save_render_path, 'mpc_state_%04d_%05d.jpg'%(seq_no, frame_no))
                bpy.context.scene.camera.location = (5, 0, 60)
                bpy.ops.render.render(write_still = True)
            elif record_state and frame_no != 1 and frame_no != endf:
                # Get the scene
                scene = bpy.context.scene
                # Set render resolution
                scene.render.resolution_x = 256
                scene.render.resolution_y = 256
                scene.render.resolution_percentage = 100
                save_render_path = os.path.join(os.getcwd(), 'mpc_policy_sa/images/s')
                bpy.context.scene.render.filepath = os.path.join(save_render_path, 'mpc_state_%04d_%05d.jpg'%(seq_no, frame_no))
                bpy.context.scene.camera.location = (5, 0, 60)
                bpy.ops.render.render(write_still = True)
            elif frame_no == endf:
                scene = bpy.context.scene
                # Set render resolution
                scene.render.resolution_x = 256
                scene.render.resolution_y = 256
                scene.render.resolution_percentage = 100
                save_render_path = os.path.join(os.getcwd(), 'mpc_policy_sa/images/s')
                bpy.context.scene.render.filepath = os.path.join(save_render_path, 'mpc_state_%04d_%05d.jpg'%(seq_no, frame_no))
                bpy.context.scene.camera.location = (5, 0, 60)
                bpy.ops.render.render(write_still = True)
    # Save the npy files
    save = os.path.join(os.getcwd(), 'mpc_policy_sa')
    np.save(os.path.join(save, 'a_spring.npy'), a)
