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

def take_action(held_link, at, keyf, settlef):
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + keyf)
    held_link.location += Vector((at[0], at[1], at[2]))
    held_link.keyframe_insert(data_path="location")
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + settlef)
    held_link.keyframe_insert(data_path="location")

if "__main__" == __name__:
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

    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)

    rope = make_rope_v3(params)
    held_link = rope[-1]
    held_link.rigid_body.kinematic = True
    init_loc = Vector(held_link.location)

    frame_end = 250 * 30
    bpy.context.scene.rigidbody_world.enabled = True
    bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end
    make_table(params)
    for seq_no in range(N):
        print('Experiment Number: ', seq_no)
        # remove all keyframes
        bpy.context.scene.frame_set(1)
        for ac in bpy.data.actions:
            bpy.data.actions.remove(ac)

        held_link.keyframe_insert(data_path="location")
        held_link.keyframe_insert(data_path="rotation_euler")

        # set up key frames
        for action_no in range(20):
            keyf = 10
            settlef = 40
            # dz needs some special handling
            dz = np.random.uniform(-held_link.location[2], 2)
            # Record the random action
            at = np.array([np.random.uniform(0.2, 2) * random.choice((-1, 1)), np.random.uniform(0.2, 2) * random.choice((-1, 1)), dz])
            print("Action taken: ", at)
            a.append(at)

            take_action(held_link, at, keyf, settlef)

        # play back animation
        endf = bpy.context.scene.frame_current
        for frame_no in range(1, endf+1):
            bpy.context.scene.frame_set(frame_no)
            record_state = (frame_no % 50 == 1)
            if record_state and frame_no == 1:
                st = []
                for r in rope:
                    st_loc = r.matrix_world.to_translation()
                    st.append(np.array(st_loc))
                s.append(st)
            elif record_state and frame_no != 1 and frame_no != endf:
                st = []
                stp1 = []
                for r in rope:
                    st_loc = r.matrix_world.to_translation()
                    st.append(np.array(st_loc))
                    stp1.append(np.array(st_loc))
                s.append(st)
                sp1.append(stp1)
            elif frame_no == endf:
                stp1 = []
                for r in rope:
                    stp1_loc = r.matrix_world.to_translation()
                    stp1.append(np.array(stp1_loc))
                sp1.append(stp1)
            # for r in rope:
                # stp1_loc = r.matrix_world.to_translation()
            # print(f"{frame_no:3d} location: {rope[0].matrix_world.to_translation()}")
    # Save the npy files
    if not os.path.exists("./states_actions"):
        os.makedirs('./states_actions')
    save = os.path.join(os.getcwd(), 'states_actions')
    np.save(os.path.join(save, 'a_spring.npy'), a)
    np.save(os.path.join(save, 's_spring.npy'), s)
    np.save(os.path.join(save, 'sp1_spring.npy'), sp1)