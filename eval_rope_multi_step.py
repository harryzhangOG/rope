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
from inv_model import *
import torch
import argparse

'''Usage: blender -P eval_rope_multi_step.py'''
"""
Multistep evaluation methadology: 
    1. Load demo states from an episode s1' ... sT', along with the demo trajectory's perturbation
    2. On a new rope, apply the same perturbation to make the starting state consistent, denote the actual starting state as st
    3. Therefore, st == st'
    4. Load the trained inverse model
    5. For t = 1 ... T - 1, run inverse model on current state st and demo's next state s_t+1', which will output a predicted action at
    6. Apply the applied action to the rope, which will give us the actual next state s_t+1
    7. Loop
"""

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
    # for step in range(start_frame, start_frame + 10):
    #     bpy.context.scene.frame_set(step)
    take_action(p2_link, start_frame + 20, (dx2, dy2, dz))
    toggle_animation(p2_link, start_frame + 20, False)

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

def eval_inv(ckpt, s1, s2):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU Cuda 0")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    inv_model = Inv_Model()
    inv_model.to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    inv_model.load_state_dict(checkpoint['model_state_dict'])
    inv_model.eval()
    # Cast to Torch tensors
    with torch.no_grad():
        s1 = torch.from_numpy(s1).to(device)
        s1 = torch.reshape(s1, (-1, 100))
        s2 = torch.from_numpy(s2).to(device)
        s2 = torch.reshape(s2, (-1, 100))
        output_action = inv_model(s1.float(), s2.float())
        return output_action
if __name__ == "__main__":

    # Checkpoint path
    ckpt = 'inv_model_ckpt.pth'
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--exp_num', dest='exp_num', type=int)
    args = parser.parse_known_args(argv)[0]
    num = args.exp_num
   

    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    clear_scene()
    rope = make_capsule_rope(params)
    rope[0].rigid_body.mass *= 5
    rope[-1].rigid_body.mass *= 2
    rig_rope(params)
    add_camera_light()
    frame_end = 3000
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end
    make_table(params) 
    # Add ref cube
    bpy.ops.mesh.primitive_cube_add(location=(10.94979668,0.53924209, 0.25), size=0.5)
    bpy.ops.rigidbody.object_add()
    ref = bpy.context.object
    ref.rigid_body.type = 'PASSIVE'
    mat = bpy.data.materials.new(name="red")
    ref.data.materials.append(mat)
    bpy.context.object.active_material.diffuse_color = (1, 0, 0, 0)    
    bpy.ops.object.select_all(action='DESELECT')
    # Perturbation taken to make s0, load the random perturbation
    pert = np.load(os.path.join(os.getcwd(), 'states_actions/multistep_pert.npy')) 
    pert2 = int(pert[0])
    dx2 = pert[1]
    dy2 = pert[2]
    # Load in series of ground truth demo states and actions
    # The states' length should be the length of an episode, T
    # Therefore, the actions' length should be the T - 1
    multistep_demo_states = np.load(os.path.join(os.getcwd(), 'states_actions/multistep_demo_states.npy'))
    multistep_demo_actions = np.load(os.path.join(os.getcwd(), 'states_actions/multistep_demo_actions.npy'))
    # for i in range(50):
    #     bpy.context.scene.frame_set(i)
    #     if i == 30:
    #         take_action(rope[-1], i, (0, 0, 0))
    #         toggle_animation(rope[-1], i, False) 
    #         take_action(rope[pert2], i, (0, 0, 0))
    #         toggle_animation(rope[pert2], i, False) 
    # # Set st in sim
    
    # If we want to perturb the rope
    perturb = 1
    if perturb: 
        set_st(pert2, dx2, dy2, 30, rope, frame_end)
    render_offset = 0
    # Iteratively predict the action
    with torch.no_grad():
        for t in range(len(multistep_demo_actions)):
            st = []
            for r in rope:
                st.append(r.matrix_world.translation[:2])
            st = np.array(st)
            # st = torch.from_numpy(st).to(device)
            # st = torch.reshape(st, (st.shape[0], -1))
            stp1_prime = multistep_demo_states[t + 1]
            # stp1_prime = torch.from_numpy(stp1_prime).to(device)
            # stp1_prime = torch.reshape(stp1_prime, (stp1_prime.shape[0], -1))
            # Use the trained inverse model to predict the action a_t
            at_pred = eval_inv(ckpt, st, stp1_prime).numpy()[0]
            print("Timestep: ", t, " Ground truth action: ", multistep_demo_actions[t], " Predicted action: ", at_pred)
            # Take the predicted action which results in actual s_t+1
            take_action(rope[-1], render_offset + 100 + at_pred[0], (at_pred[1], at_pred[2], 0))
            for i in range(render_offset, render_offset + 200):
                bpy.context.scene.frame_set(i)
                if i == 50 and perturb:
                    save_render_path = os.path.join(os.getcwd(), 'inv_model_15k_multistep')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'pred_perturb_exp_%d.jpg'%(num))
                    bpy.context.scene.camera.location = (0, 0, 60)
                    bpy.ops.render.render(write_still = True)
                if i % 10 == 0:
                    save_render_path = os.path.join(os.getcwd(), 'inv_model_15k_multistep/video/pred')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'exppred_%d_frame_%03d.jpg'%(num, i))
                    bpy.context.scene.camera.location = (0, 0, 60)
                    bpy.ops.render.render(write_still = True)

            render_offset += 200
            save_render_path = os.path.join(os.getcwd(), 'inv_model_15k_multistep')
            bpy.context.scene.render.filepath = os.path.join(save_render_path, 'pred_exp_%d_%d.jpg'%(num, t))
            bpy.context.scene.camera.location = (0, 0, 60)
            bpy.ops.render.render(write_still = True)

        # Evaluate terminal state error
        gt_free_end = multistep_demo_states[-1, 0, :]
        pred_free_end = np.array(rope[0].matrix_world.translation)[:2]
        print("Ground truth rope free end location: ", gt_free_end)
        print("Predicted rope free end location: ", pred_free_end)
        # Calculate error
        mse_diff = np.linalg.norm(gt_free_end - pred_free_end)**2
        print("Terminal state free end MSE: %03f" %(mse_diff))
        
    

