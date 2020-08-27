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
from ur5_viz import UR5, add_child
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
sys.path.append('/Users/harryzhang/Library/Python/3.7/lib/python/site-packages')
from cvxopt import spmatrix, matrix, solvers, printing

# Sparse matrices. Collects rows, columns, and values as triples
# to be later passed to the spmatrix function.
class SPMatBuilder:
    def __init__(self):
        self.v = []
        self.i = []
        self.j = []
        #self._cell_map = {}

    def append(self, r, c, s):
        if abs(s) > 1e-6:
            #assert (r,c) not in self._cell_map
            self.i.append(r)
            self.j.append(c)
            self.v.append(s)
            #self._cell_map[(r,c)] = len(self.i) - 1

    def build(self, rows, cols):
        return spmatrix(self.v, self.i, self.j, (rows, cols), 'd')

MIN_CONFIG = np.array([ -2*pi, -pi, -2*pi, -2*pi, -2*pi, -2*pi ])
MAX_CONFIG = np.array([  2*pi,   0,  2*pi,  2*pi,  2*pi,  2*pi ])

MAX_VELOCITY = np.array([3.]*6)
MIN_VELOCITY = -MAX_VELOCITY

MAX_ARM_EFFORT = 150.
MAX_WRIST_EFFORT = 28.

# UR5 has a payload limit of 5 kg
MASS_LIMIT = 0.5

MAX_EXTENSION = np.array([ 1.25, 1.25, 0.75, 0.4, 0.3, 0.2 ])

# Compute an approximation of the maximum acceleration based on
# holding the maximum payload at the maximum extension
MAX_ACCELERATION = np.array([
    MAX_ARM_EFFORT   / (MASS_LIMIT * 1.25), # shoulder pan
    MAX_ARM_EFFORT   / (MASS_LIMIT * 1.25), # shoulder lift
    MAX_ARM_EFFORT   / (MASS_LIMIT * 0.75), # elbow
    MAX_WRIST_EFFORT / (MASS_LIMIT * 0.4), # wrist 1
    MAX_WRIST_EFFORT / (MASS_LIMIT * 0.3), # wrist 2
    MAX_WRIST_EFFORT / (MASS_LIMIT * 0.2) # wrist 3
    ])

MAX_JERK = MAX_ACCELERATION

def generate_whip_motion(start_config, end_config, H, t_step):
    # Build QP in form:
    #
    # min_x 1/2 x^T P x + q^T x
    #   s.t. Ax = b
    #        Gx < h
    #
    h, b = [], []
    P = SPMatBuilder()
    A = SPMatBuilder()
    G = SPMatBuilder()

    mid = (H+1)//2
    mid_angle = pi/3.
        
    dim = 6
    num_states = (H+1)*dim
    a0_index = num_states * 2
    
    def qvar(t, j): return t*dim + j
    def vvar(t, j): return num_states + t*dim + j
    def avar(t, j): return a0_index + t*dim + j

    ssa = 1. / (t_step * H)
    for t in range(0, H+1):
        for j in range(dim):
            P.append(avar(t, j), avar(t, j), MAX_EXTENSION[j]**2 * ssa)

    # joint ranges
    for t in range(1, H):
        for j in range(dim):
            # G.append(len(h), qvar(t, j), 1.)
            # h.append( MAX_CONFIG[j])
            # G.append(len(h), qvar(t, j), -1.)
            # h.append(-MIN_CONFIG[j])

            if t == mid and j == 2:
                # 0 velocity at mid
                A.append(len(b), vvar(t,j), 1.)
                b.append(0.)
            else:
                G.append(len(h), vvar(t, j), 1.)
                h.append( MAX_VELOCITY[j])
                G.append(len(h), vvar(t, j), -1.)
                h.append( MAX_VELOCITY[j])

            G.append(len(h), avar(t, j), 1.)
            h.append( MAX_ACCELERATION[j])
            G.append(len(h), avar(t, j), -1.)
            h.append( MAX_ACCELERATION[j])

    if True:
        # jerk limits
        # jerk = (a_{t+1} - a_t) / t_step
        for t in range(H+1):
            for j in range(dim):
                if t > 0:
                    G.append(len(h),   avar(t-1, j),  1./t_step)
                    G.append(len(h)+1, avar(t-1, j), -1./t_step)
                if t <= H:
                    G.append(len(h),   avar(t, j), -1./t_step)
                    G.append(len(h)+1, avar(t, j),  1./t_step)
                h.append(MAX_JERK[j])
                h.append(MAX_JERK[j])

    # set up counter rotations
    for t in range(H+1):
        # q[0] == q[4] - pi/2
        if False:
            # this keeps the gripper pointing in the same direction
            A.append(len(b), qvar(t, 0),  1.)
            A.append(len(b), qvar(t, 4), -1.)
            b.append(-pi/2.)
        else:
            # this co-rotates for more acceleration
            # q[4] = pi/2 + q[0]*0.25
            A.append(len(b), qvar(t, 0),  0.25)
            A.append(len(b), qvar(t, 4), 1.0)
            b.append(pi/2.)

        # co-rotate elbow with shoulder-left
        # -pi*2/12 and pi/3 #pi/2 - pi/6
        # -pi*3/12 and 0
        # -pi*4/12 and -pi/3
        #A.append(len(b), qvar(t, 1), 1.)
        #A.append(len(b), qvar(t, 2), 1.)
        #b.append(0.)

        if False:
            # keep the end effector level
            # q[1] + q[2] + q[3] == pi/4
            A.append(len(b), qvar(t, 1), 1.)
            A.append(len(b), qvar(t, 2), 1.)
            A.append(len(b), qvar(t, 3), 1.)
            b.append(0.) #pi/4)
        else:
            # Have q[3] == q[2]
            A.append(len(b), qvar(t, 1), 1.)
            A.append(len(b), qvar(t, 2), 1.)
            A.append(len(b), qvar(t, 3), -1.)
            b.append(0.)

    # Have the mid point reach mid_angle
    #A.append(len(b), qvar(mid, 2), 1.)
    A.append(len(b), qvar(mid, 1), -1.)
    A.append(len(b), qvar(mid, 2), -0.5)
    b.append(mid_angle)

    # dynamics constraints
    for t in range(H):
        for j in range(dim):
            if j == 4 or j == 3:
                continue
            A.append(len(b), qvar(t+1, j), -1.0/t_step)
            A.append(len(b), qvar(t,   j),  1.0/t_step)
            A.append(len(b), vvar(t,   j),  1.0)
            A.append(len(b), avar(t,   j),  t_step/3.)
            A.append(len(b), avar(t+1, j),  t_step/6.)
            b.append(0.)

            A.append(len(b), vvar(t+1, j), -1.0)
            A.append(len(b), vvar(t,   j),  1.0)
            A.append(len(b), avar(t,   j),  t_step/2.)
            A.append(len(b), avar(t+1, j),  t_step/2.)
            b.append(0.)

    for j in range(dim):
        # initial conditions
        if j != 4 and j != 3:
            A.append(len(b), qvar(0, j), 1.)
            b.append(start_config[j])
        A.append(len(b), vvar(0, j), 1.)
        b.append(0.)
        A.append(len(b), avar(0, j), 1.)
        b.append(0.)

        # terminal conditions
        if j != 4 and j != 3:
            A.append(len(b), qvar(H, j), 1.)
            b.append(end_config[j])
        A.append(len(b), vvar(H, j), 1.)
        b.append(0.)
        A.append(len(b), avar(H, j), 1.)
        b.append(0.)

    n = num_states*3
    # convert temporary structures to format cvxopt prefers
    P = P.build(n, n)
    A = A.build(len(b), n)
    G = G.build(len(h), n)
    q = matrix(np.zeros(n))
    b = matrix(np.array(b))
    h = matrix(np.array(h))

    sol = solvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal':
        return None

    traj = np.array(sol['x']).reshape((3*(H+1),6))[0:H+1,:]
    return traj

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
    ur5.base.scale[0] = 4
    ur5.base.scale[1] = 4
    ur5.base.scale[2] = 4
    ur5.base.location[0] = 5
    ur5.base.location[2] = 2
    bpy.context.view_layer.objects.active = ur5.base
    bpy.context.object.rotation_mode = 'XYZ'
    ur5.base.rotation_euler = Euler((0, 0, pi), 'XYZ')
    bpy.ops.object.transform_apply(location = False, scale = True, rotation = False)
    bpy.context.view_layer.update()

    start_config = [-pi/4., 0., pi/6., -pi/4, pi/4., 0.]
    end_config   = [ pi/4., -pi/6., pi/2 - pi/4., -pi/4, pi/4. + pi/2, 0.]
    duration = 1.2 # seconds
    fps = 24
    H = ceil(fps*duration)
    traj = generate_whip_motion(start_config, end_config, H, 1./fps)
    while True:
        H = H - 1
        traj_new = generate_whip_motion(start_config, end_config, H, 1./fps)
        if traj_new is None:
            break
        traj = traj_new

    kf = 51
    ur5.set_config(start_config)
    ur5.keyframe_insert(1)
    bpy.context.scene.frame_set(1)

    # 2a. A keyframe motion of held link to put in gripper (start kf = 1, end kf = 50)
    held_link.keyframe_insert(data_path="location")
    held_link.keyframe_insert(data_path="rotation_euler")
    bpy.context.scene.rigidbody_world.enabled = True
    bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
    target_loc = (ur5.gripper.right_inner_finger_pad.matrix_world.translation + ur5.gripper.left_inner_finger_pad.matrix_world.translation)/2
    at = target_loc - held_link.matrix_world.translation
    print(at)
    take_action(held_link, at, keyf, 50-keyf)
    ur5.keyframe_insert(keyf)
    # # 2b. Set keyframe to ur5
    ur5.keyframe_insert(51)
    for i in range(1, 52):
        bpy.context.scene.frame_set(i)

    # 3. Add rigid-body constraint to attach held end to ur5.gripper.gripper_base
    # Inverse the transform matrix to make the transform correct
    held_link.parent = ur5.gripper.right_inner_finger_pad
    held_link.matrix_parent_inverse = ur5.gripper.right_inner_finger_pad.matrix_world.inverted()

    # 4. make held link.kinematic=False
    held_link.rigid_body.kinematic = True
    # 5. ur5.set_config(...), insert keyframes
    for t in range(traj.shape[0]):
        wp = traj[t,:]
        print(wp)
        ur5.set_config(wp)
        ur5.keyframe_insert(kf)
        kf = kf + 1

    for i in range(51, 100):
        bpy.context.scene.frame_set(i)



