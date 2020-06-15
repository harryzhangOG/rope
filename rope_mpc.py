"""
Minimal working MPC via rope's forward dynamics model.

Model Predictive Control:
    - Model: A deep neural network trained on 50k rope interactions.
             f(s, a) -> s'
    - Prediction horizon: N
    - Terminal state: when t + N >= T, where T = 5/10
    - Cost function: Minimizing the L2 distance between the rope free link's locations in predicted motion and demo (gt) motion.
    - In each prediction iteration:
      * Based on current s_t and iteratively apply f on s_t sampled a_t up to N steps ahead, calculate d(s_{t+N}, s_{t+N}')
      * Rollout a_t that minimizes d(s_{t+1}, s_{t+1}'), take one step in Blender
      * Repeat

Usage: blender -P rope_mpc.py -- -N PRED_HORIZON -exp EXP_NUM
"""

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
from fwd_model import *
import torch
import argparse

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

def load_fwd(ckpt):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU Cuda 0")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    fwd_model = Fwd_Model()
    fwd_model.to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    fwd_model.load_state_dict(checkpoint['model_state_dict'])
    return fwd_model

def eval_fwd(fwd_model, st, at):
    fwd_model.eval()
    # Cast to Torch tensors
    with torch.no_grad():
        st = torch.from_numpy(st).to(device)
        st = torch.reshape(st, (-1, 100))
        at = torch.from_numpy(at).to(device)
        at = torch.reshape(at, (-1, 3))
        output_stp1 = fwd_model(st.float(), stp1.float())
        return output_stp1

def mpc(fwd_model, st, stpN_prime, T, N, residual):
    """
    Given current state s_t and demo state s_{t+N}', calculate the sequence {a_t, ..., a_{t+N-1}}.
    Return a_t
    """
    # Define cost function as the squared L2 diff between gt stpN_prime and predicted stpN
    cost = lambda stpN: np.linalg.norm(stpN[0, :] - stpN_prime[0, :])**2
    # Allocate cost values map
    c_val = {}
    # Optimization loop
    min_cost = 10_000
    min_actions = None
    # TODO: Better sampling method, currently random search
    for i in range (100):
        actions = []
        # Plan ahead N steps. If terminal, plan ahead residual steps
        if residual < N:
            for t in range(residual - 1):
                at = np.array([np.random.uniform(0.2, 2) * random.choice((-1, 1)), np.random.uniform(0.2, 2) * random.choice((-1, 1)), np.random.uniform(0.2, 2) * random.choice((-1, 1))])
                st = eval_fwd(fwd_model, st, at)
                actions.append(at)
        else:    
            for t in range(N - 1):
                at = np.array([np.random.uniform(0.2, 2) * random.choice((-1, 1)), np.random.uniform(0.2, 2) * random.choice((-1, 1)), np.random.uniform(0.2, 2) * random.choice((-1, 1))])
                st = eval_fwd(fwd_model, st, at)
                actions.append(at)
        # Calculate current cost-to-go for comparison online
        cost_to_go = cost(st)
        if cost_to_go < min_cost:
            min_cost = cost_to_go
            min_actions = actions
    return min_cost, min_actions[0]


if __name__ == "__main__":

    # Checkpoint path
    ckpt = 'fwd_model_ckpt_3d.pth'
    # Load forward model aka system model
    fwd_model = load_fwd(ckpt)

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--pred_hrzn', dest='pred_hrzn', type=int)
    parser.add_argument('-exp', '--exp_num', dest='exp_num', type=int)
    args = parser.parse_known_args(argv)[0]
    # Prediction horizon N
    N = args.pred_hrzn
    # Experiment number
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

    # Load in series of ground truth demo states and actions
    # The states' length should be the length of an episode, T
    # Therefore, the actions' length should be the T - 1
    multistep_demo_states = np.load(os.path.join(os.getcwd(), 'states_actions/multistep_demo_states.npy'))
    multistep_demo_actions = np.load(os.path.join(os.getcwd(), 'states_actions/multistep_demo_actions.npy'))

    # Horizon
    T = len(multistep_demo_states)

    # Add ref cube
    terminal_link_x = multistep_demo_states[-1, 0, 0]
    terminal_link_y = multistep_demo_states[-1, 0, 1]
    bpy.ops.mesh.primitive_cube_add(location=(terminal_link_x, terminal_link_y, 0.5), size=0.5)
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
    
    # If we want to perturb the rope
    perturb = 0
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

            residual = T - t
            if residual <= N: 
                stpN = multistep_demo_states[-1]
            else:
                stpN_prime = multistep_demo_states[t + N]
            # Use MPC based on trained fwd model to predict the action a_t
            min_ctg, at_pred = mpc(fwd_model, st, stpN_prime, T, N, residual)
            print("Timestep: ", t, " Ground truth action: ", multistep_demo_actions[t], " Predicted action: ", at_pred, " Min cost-to-go", min_ctg)
            
            # Take the predicted action which results in actual s_t+1, if PERTURB, need 100 frame to buffer
            if perturb: 
                take_action(rope[-1], render_offset + 110, (at_pred[0], at_pred[1], at_pred[2]))
            else:
                take_action(rope[-1], render_offset + 10, (at_pred[0], at_pred[1], at_pred[2]))

            for i in range(render_offset, render_offset + 100):
                bpy.context.scene.frame_set(i)
                if i == 50 and perturb:
                    save_render_path = os.path.join(os.getcwd(), 'inv_model_50k_multistep')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'pred_perturb_exp_%d.jpg'%(num))
                    bpy.context.scene.camera.location = (0, 0, 60)
                    bpy.ops.render.render(write_still = True)
                if i % 10 == 0:
                    save_render_path = os.path.join(os.getcwd(), 'inv_model_50k_multistep_mpc/video/pred')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'exppred_%d_frame_%03d.jpg'%(num, i))
                    bpy.context.scene.camera.location = (0, 0, 60)
                    bpy.ops.render.render(write_still = True)

            render_offset += 100
            save_render_path = os.path.join(os.getcwd(), 'inv_model_50k_multistep')
            bpy.context.scene.render.filepath = os.path.join(save_render_path, 'pred_exp_%d_%d.jpg'%(num, t))
            bpy.context.scene.camera.location = (0, 0, 60)
            bpy.ops.render.render(write_still = True)