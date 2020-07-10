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

Usage: blender -P rope_mpc.py -- -N PRED_HORIZON -exp EXP_NUM -render RENDER_BOOLEAN
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

def take_action(held_link, at, keyf, settlef):
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + keyf)
    held_link.location += Vector((at[0], at[1], at[2]))
    held_link.keyframe_insert(data_path="location")
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + settlef)
    held_link.keyframe_insert(data_path="location")

# def take_action(obj, frame, action_vec, animate=True):
#     # Keyframes a displacement for obj given by action_vec at given frame
#     curr_frame = bpy.context.scene.frame_current
#     dx,dy,dz = action_vec
#     if animate != obj.rigid_body.kinematic:
#         # We are "picking up" a dropped object, so we need its updated location
#         obj.location = obj.matrix_world.translation
#         obj.rotation_euler = obj.matrix_world.to_euler()
#         obj.keyframe_insert(data_path="location", frame=curr_frame)
#         obj.keyframe_insert(data_path="rotation_euler", frame=curr_frame)
#     toggle_animation(obj, curr_frame, animate)
#     obj.location += Vector((dx,dy,dz))
#     obj.keyframe_insert(data_path="location", frame=frame)

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
    return device, fwd_model

def eval_fwd(fwd_model, st, at, device):
    fwd_model.eval()
    # Cast to Torch tensors
    with torch.no_grad():
        st = torch.from_numpy(st).to(device)
        st = torch.reshape(st, (-1, 270))
        at = torch.from_numpy(at).to(device)
        at = torch.reshape(at, (-1, 3))
        output_stp1 = fwd_model(st.float(), at.float())
        return output_stp1

def mpc(rope, fwd_model, st, stpN_prime, T, N, residual, device, CEM=True):
    """
    Given current state s_t and demo state s_{t+N}', calculate the sequence {a_t, ..., a_{t+N-1}}.
    Return a_t
    """

    # Define cost function as the squared L2 diff between gt stpN_prime and predicted stpN
    cost = lambda t, stpN: np.linalg.norm(stpN - stpN_prime[t, :, :].reshape((-1, 100)))**2
    # cost = lambda t, stpN: np.linalg.norm(stpN[:2] - stpN_prime[t, 0, :])**2

    # Allocate cost values map
    c_val = {}

    # Optimization loop
    min_cost = 10_000
    min_actions = None

    # CEM Sampling
    def action_cem(st, num_elite, cost, max_iter, alpha=0.9, epsilon=1e-3, opt_mode="MAX_ITER"):
        """
        Cross entropy method for action sampling in the optimization step.
        We model the distribution of action as a normal distribution
        - ST: Current state
        - MU: Initial expectation of the actions in the format of [mu_x, mu_y, mu_z].
        - STD: Initial standard deviation of the actions in the format of [sigma_x, sigma_y, sigma_z]
        - NUM_ELITE: Number of elites actions that we are refitting to.
        - COST: Cost function metric to rank the elite actions.
        - MAX_ITER: Maximum number of optimization steps.
        - ALPHA: Polyak averaging parameter
        - EPSILON: Minimum value for std
        - OPT_MODE: If we are using Epsilon or Max_iter
        """
        it = 0
        cost_queue = []
        actions_queue = []

        # Max iteration stopping condition
        if opt_mode == "MAX_ITER":
            if residual < N:
                steps = residual
            else:
                steps = N
            # Save current state to restore
            st_save = st

            for _ in range(100):
                cost_val = 0
                actions = []
                # Restore st
                st = st_save

                for t in range(steps):
                    # Inside the while loop is CEM step.
                    while it < max_iter:
                        # Generate action candidates
                        if it == 0:
                            actions_candidates = [[np.random.uniform(0.2, 2) * random.choice((-1, 1)), 
                                            np.random.uniform(0.2, 2) * random.choice((-1, 1)), 
                                            np.random.uniform(-rope[-1].matrix_world.translation[2], 2)] for _ in range(1000)]

                            mu_pre = np.array(actions_candidates).mean(axis=0)
                            std_pre = np.array(actions_candidates).std(axis=0)
                        else:
                            actions_candidates = [[mu[0] + std[0]*np.random.randn(), mu[1] + std[1]*np.random.randn(), mu[2] + std[2]*np.random.randn()] for _ in range(100)]

                        # Evaluate the candidates by rolling each action out in sim
                        cost_to_go = [cost(t, eval_fwd(fwd_model, st, np.array(at), device).numpy()[0]) for at in actions_candidates]

                        # Find elite actions according to the cost_to_go function
                        if it == 0:
                            elite_idxs = np.array(cost_to_go).argsort()[:200]
                            elite_actions = [actions_candidates[idx] for idx in elite_idxs]
                        else:
                            elite_idxs = np.array(cost_to_go).argsort()[:num_elite]
                            elite_actions = [actions_candidates[idx] for idx in elite_idxs]


                        # Refit the distribution
                        mu = np.array(elite_actions).mean(axis=0)
                        std = np.array(elite_actions).std(axis=0)

                        # Polyak Averaging
                        mu = alpha * mu + (1 - alpha) * mu_pre
                        std = alpha * std + (1 - alpha) * std_pre
                        mu_pre = mu
                        std_pre = std

                        # Next optimization step
                        it += 1

                    # Find current time step's minimal ctg
                    min_ctg = cost_to_go[elite_idxs[0]]
                    cost_val += min_ctg
                    # Sample an elite action
                    min_at = np.array([mu[0] + std[0]*np.random.randn(), mu[1] + std[1]*np.random.randn(), mu[2] + std[2]*np.random.randn()])
                    actions.append(min_at)
                    st = eval_fwd(fwd_model, st, min_at, device).numpy()[0]

                cost_queue.append(cost_val)
                actions_queue.append(actions)
            
            # Return min ctg and action
            min_cost = np.min(np.array(cost_queue))
            min_actions_seq = np.array(actions_queue)[np.array(cost_queue).argmin(), :, :]
            print(cost_queue)

            return min_cost, min_actions_seq[0, :]

        # Std epsilon stopping condition
        elif opt_mode == "EPSILON":
            if residual < N:
                steps = residual
            else:
                steps = N
            # Save current state to restore
            st_save = st

            for _ in range(100):
                cost_val = 0
                actions = []
                # Restore st
                st = st_save

                for t in range(steps):
                    # Inside the while loop is CEM step.
                    std = np.array([1, 1, 1])
                    while np.max(std) > epsilon:
                        # Generate action candidates
                        if it == 0:
                            actions_candidates = [[np.random.uniform(0.2, 2) * random.choice((-1, 1)), 
                                            np.random.uniform(0.2, 2) * random.choice((-1, 1)), 
                                            np.random.uniform(-rope[-1].matrix_world.translation[2], 1)] for _ in range(1000)]
                            mu_pre = np.array([0, 0, 0])
                            std_pre = np.array([1, 1, 1])
                        else:
                            actions_candidates = [[mu[0] + std[0]*np.random.randn(), mu[1] + std[1]*np.random.randn(), mu[2] + std[2]*np.random.randn()] for _ in range(1000)]

                        # Evaluate the candidates by rolling each action out in sim
                        cost_to_go = [cost(t, eval_fwd(fwd_model, st, np.array(at), device).numpy()[0]) for at in actions_candidates]

                        # Find elite actions according to the cost_to_go function
                        elite_idxs = np.array(cost_to_go).argsort()[:num_elite]
                        elite_actions = [actions_candidates[idx] for idx in elite_idxs]

                        # Refit the distribution
                        mu = np.array(elite_actions).mean(axis=0)
                        std = np.array(elite_actions).std(axis=0)

                        # Polyak Averaging
                        mu = alpha * mu + (1 - alpha) * mu_pre
                        std = alpha * std + (1 - alpha) * std_pre
                        mu_pre = mu
                        std_pre = std

                        # Next optimization step
                        it += 1

                    # Find current time step's minimal ctg
                    min_ctg = cost_to_go[elite_idxs[0]]
                    cost_val += min_ctg
                    # Sample an elite action
                    min_at = np.array([mu[0] + std[0]*np.random.randn(), mu[1] + std[1]*np.random.randn(), mu[2] + std[2]*np.random.randn()])
                    actions.append(min_at)
                    st = eval_fwd(fwd_model, st, min_at, device).numpy()[0]

                cost_queue.append(cost_val)
                actions_queue.append(actions)
            
            # Return min ctg and action
            min_cost = np.min(np.array(cost_queue))
            min_actions_seq = actions_queue[np.array(cost_queue).argmin(), :, :]

            return min_cost, min_actions_seq[0, :]

    if not CEM: 
        for i in range (10_000):
            actions = []
            # Plan ahead N steps. If terminal, plan ahead residual steps
            cost_to_go = 0
            if residual < N:
                for t in range(residual):
                    dz = np.random.uniform(-rope[-1].matrix_world.translation[2], 2)
                    at = np.array([np.random.uniform(0.2, 2) * random.choice((-1, 1)), np.random.uniform(0.2, 2) * random.choice((-1, 1)), dz])
                    st = eval_fwd(fwd_model, st, at, device).numpy()[0]
                    actions.append(at)
                    # Calculate current cost-to-go for comparison online
                    cost_to_go += cost(t, st)
            else:    
                for t in range(N):
                    dz = np.random.uniform(-rope[-1].matrix_world.translation[2], 2)
                    at = np.array([np.random.uniform(0.2, 2) * random.choice((-1, 1)), np.random.uniform(0.2, 2) * random.choice((-1, 1)), dz])
                    st = eval_fwd(fwd_model, st, at, device).numpy()[0]
                    actions.append(at)
                    # Calculate current cost-to-go for comparison online
                    cost_to_go += cost(t, st)
            if cost_to_go < min_cost:
                min_cost = cost_to_go
                min_actions = actions
        return min_cost, min_actions[0]
    else:
        return action_cem(st, 20, cost, 20)
         

def dagger(st, at, stp1, s_dataset, a_dataset, sp1_dataset): 
    """
    Dataset aggregation procedure for evaluation.
    Saves (st, at, stp1) pair to path npy file.
    In this case, stp1 is our label that we manually record. We discard the predicted stp1.
    """

    # Aggregate the new datapoints
    st = st.reshape((-1, 90, 3))
    s_dataset = np.vstack((s_dataset, st))
    stp1 = stp1.reshape((-1, 90, 3))
    sp1_dataset = np.vstack((sp1_dataset, stp1))
    a_dataset = np.vstack((a_dataset, at))

    return s_dataset, a_dataset, sp1_dataset    


if __name__ == "__main__":
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)

    clear_scene()
    rope = make_rope_v3(params)
    make_table(params)
    bpy.context.scene.gravity *= 10
    add_camera_light()

    # Checkpoint path
    ckpt = 'fwd_model_ckpt_3d_spring.pth'
    # Load forward model aka system model
    device, fwd_model = load_fwd(ckpt)
    # If we want to aggregate the dataset
    dagger_ind = False

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--pred_hrzn', dest='pred_hrzn', type=int)
    parser.add_argument('-exp', '--exp_num', dest='exp_num', type=int)
    parser.add_argument('-render', '--render', dest='render', type=int)

    args = parser.parse_known_args(argv)[0]
    # Prediction horizon N
    N = args.pred_hrzn
    # Experiment number
    num = args.exp_num
    # Render flag
    render = args.render
    
    held_link = rope[-1]
    held_link.rigid_body.kinematic = True
    
    frame_end = 250 * 30
    bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_end
    bpy.context.scene.frame_end = frame_end

    for r in rope:
        r.rigid_body.mass *= 10
    rope[0].rigid_body.mass *= 5
    
    # Load in series of ground truth demo states and actions
    # The states' length should be the length of an episode, T
    # Therefore, the actions' length should be the T - 1
    multistep_demo_states = np.load(os.path.join(os.getcwd(), 'states_actions/test/multistep_demo_states_%d.npy'%(num)))
    multistep_demo_actions = np.load(os.path.join(os.getcwd(), 'states_actions/test/multistep_demo_actions_%d.npy'%(num)))

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

    # If we want to perturb the rope
    perturb = 0
    if perturb: 
        # Perturbation taken to make s0, load the random perturbation
        pert = np.load(os.path.join(os.getcwd(), 'states_actions/multistep_pert.npy')) 
        pert2 = int(pert[0])
        dx2 = pert[1]
        dy2 = pert[2]

        set_st(pert2, dx2, dy2, 30, rope, frame_end)
    render_offset = 0

    # Load datasets
    path = os.path.join(os.getcwd(), "states_actions")
    s_dataset = np.load(os.path.join(path, 's_spring.npy'))
    a_dataset = np.load(os.path.join(path, 'a_spring.npy'))
    sp1_dataset = np.load(os.path.join(path, 'sp1_spring.npy'))

    # Iteratively predict the action
    with torch.no_grad():
        bpy.context.scene.frame_set(1)
        for ac in bpy.data.actions:
            bpy.data.actions.remove(ac)

        held_link.keyframe_insert(data_path="location")
        held_link.keyframe_insert(data_path="rotation_euler")
        bpy.context.scene.rigidbody_world.enabled = True
        bpy.context.scene.rigidbody_world.point_cache.frame_start = 1

        for t in range(T):
            st = []
            for r in rope:
                st.append(r.matrix_world.translation)
            st = np.array(st)

            residual = T - t
            if residual <= N: 
                stpN_prime = multistep_demo_states[t:]
            else:
                stpN_prime = multistep_demo_states[t:t+N+1]
            # Use MPC based on trained fwd model to predict the action a_t
            min_ctg, at_pred = mpc(rope, fwd_model, st, stpN_prime, T, N, residual, device, True)
            print("Timestep: ", t, " Ground truth action: ", multistep_demo_actions[t], " Predicted action: ", at_pred, " Min cost-to-go", min_ctg)
            
            # Take the predicted action which results in actual s_t+1, if PERTURB, need 100 frame to buffer
            if perturb: 
                take_action(rope[-1], (at_pred[0], at_pred[1], at_pred[2]), 110, 40)
            else:
                take_action(rope[-1], (at_pred[0], at_pred[1], at_pred[2]), 10, 40)

            for i in range(render_offset + 1, render_offset + 51):
                bpy.context.scene.frame_set(i)
                if i == 50 and perturb and render:
                    save_render_path = os.path.join(os.getcwd(), 'fwd_model_6k_spring')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'pred_perturb_exp_%d.jpg'%(num))
                    bpy.context.scene.camera.location = (0, 0, 60)
                    bpy.ops.render.render(write_still = True)
                if i % 10 == 0 and render:
                    save_render_path = os.path.join(os.getcwd(), 'fwd_model_6k_spring/video/pred')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'exppred_%d_frame_%03d.jpg'%(num, i))
                    bpy.context.scene.camera.location = (0, 0, 60)
                    bpy.ops.render.render(write_still = True)
            stp1 = []
            for r in rope:
                stp1.append(r.matrix_world.translation)
            stp1 = np.array(stp1)

            # Aggregate the dataset every time step.
            if dagger_ind:
                s_dataset, a_dataset, sp1_dataset = dagger(st, at_pred, stp1, s_dataset, a_dataset, sp1_dataset)

            render_offset += 50

            if render:
                save_render_path = os.path.join(os.getcwd(), 'fwd_model_6k_mpc')
                bpy.context.scene.render.filepath = os.path.join(save_render_path, 'pred_exp_%d_%d.jpg'%(num, t))
                bpy.context.scene.camera.location = (0, 0, 60)
                bpy.ops.render.render(write_still = True)

    # Write the new datapoints to the dataset
    np.save(os.path.join(path, 's_spring.npy'), s_dataset)
    np.save(os.path.join(path, 'a_spring.npy'), a_dataset)
    np.save(os.path.join(path, 'sp1_spring.npy'), sp1_dataset)

    # Evaluate final MSE
    # Evaluate terminal state error
    gt_free_end = multistep_demo_states[-1, 0, :]
    pred_free_end = np.array(rope[0].matrix_world.translation)[:2]
    print("Ground truth rope free end location: ", gt_free_end)
    print("Predicted rope free end location: ", pred_free_end)
    # Calculate error
    mse_diff = np.linalg.norm(gt_free_end - pred_free_end)**2
    print("Terminal state free end MSE: %03f" %(mse_diff))