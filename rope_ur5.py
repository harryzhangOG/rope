import bpy
import os
import json
import time
import sys
import bpy, bpy_extras
from math import *
from mathutils import *
import random
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
#sys.path.append('/Users/harryzhang/Library/Python/3.7/lib/python/site-packages')
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import numpy as np
sys.path.append('/Users/harryzhang/Library/Python/3.7/lib/python/site-packages')
from train_ur5_sim_resnet import DistModel
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
MAX_JERK[1] *= 0.75

def qp_whip_motion(start_config, mid_config, end_config, H, t_step):
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

    # setup the minimization objective:
    # sub-of-squared accelerations
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

            if t == mid and (j == 1 or j == 2 or j == 3):
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
    if False:
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
                A.append(len(b), qvar(t, 3), -2)
                b.append(-pi)

    # # Have the mid point reach mid_angle
    # #A.append(len(b), qvar(mid, 2), 1.)
    # A.append(len(b), qvar(mid, 1), -1.)
    # A.append(len(b), qvar(mid, 2), -0.5)
    # b.append(mid_angle)

    # dynamics constraints
    for t in range(H):
        for j in range(dim):
            #if j == 4 or j == 3:
            #    continue
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
        #if j != 4 and j != 3:
        A.append(len(b), qvar(0, j), 1.)
        b.append(start_config[j])
        A.append(len(b), vvar(0, j), 1.)
        b.append(0.)
        A.append(len(b), avar(0, j), 1.)
        b.append(0.)

        A.append(len(b), qvar(mid, j), 1.)
        b.append(mid_config[j])

        # terminal conditions
        #if j != 4 and j != 3:
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
        return None, None, None

    traj = np.array(sol['x']).reshape((3*(H+1),6))
    return traj[0:H+1,:], traj[H+1:(H+1)*2], traj[(H+1)*2:(H+1)*3]

def generate_whip_motion(start_config, mid_config, end_config, H, t_step):
    cfg, vel, acc = qp_whip_motion(start_config, mid_config, end_config, H, t_step)
    while True:
        H = H - 1
        cnew, vnew, anew = qp_whip_motion(start_config, mid_config, end_config, H, t_step)
        if cnew is None:
            return cfg, vel, acc, H
        cfg, vel, acc = cnew, vnew, anew

def bpy_main(start_config, mid_config, end_config):
    import bpy
    from ur5_viz import UR5
    # Blender always starts with a cube, delete it.
    bpy.ops.object.delete(use_global=False)

    #start_config = [-pi/3., -pi/6., pi/3, -pi/4, pi/4., 0.]
    # start_config = [-pi/4., 0., pi/6., -pi/4, pi/4., 0.]
    # end_config   = [ pi/4., -pi/6., pi/2 - pi/4., -pi/4, pi/4. + pi/2, 0.]
    # start_config = [-pi/4., -pi      , -pi/6., -pi/4, pi/4., 0.]
    # end_config   = [ pi/4., pi/6. -pi, -pi/6., -pi/4, pi/4. + pi/2, 0.]
    duration = 1.2 # seconds
    fps = 24
    H = ceil(fps*duration)
    cfg, vel, acc, H = generate_whip_motion(start_config, mid_config, end_config, H, 1./fps)
    
    ur5 = UR5()
    kf = 1
    for t in range(cfg.shape[0]):
        wp = cfg[t,:]
        # print(wp)
        ur5.set_config(wp)
        ur5.keyframe_insert(kf)
        kf = kf + 1
        
    bpy.context.scene.frame_end = kf
    bpy.context.scene.frame_set(1)

    bpy.ops.mesh.primitive_cube_add(
        size=2, enter_editmode=False, align='WORLD',
        location=(0.8, 0, -1), rotation=(0, 0, 0))

    camera = bpy.data.objects['Camera']
    camera.location = (-4.031968116760254, -0.6357017755508423, 2.1259286403656006)
    camera.rotation_euler = (1.1429533958435059, 3.1036906875669956e-06, -1.4192959070205688)
    light = bpy.data.objects['Light']
    light.location = (-3.047271728515625, -1.5726829767227173, 5.903861999511719)

def dat_main(start_config, mid_config, end_config):
    # start_config = [-pi/4., 0., pi/6., -pi/4, pi/4., 0.]
    # end_config   = [ pi/4., -pi/6., pi/2 - pi/4., -pi/4, pi/4. + pi/2, 0.]
    #start_config = [-pi/4., -pi      , -pi/6., -pi/4, pi/4., 0.]
    #end_config   = [ pi/4., pi/6. -pi, -pi/6., -pi/4, pi/4. + pi/2, 0.]

    duration = 2 # seconds
    t_step = 0.032
    H = ceil(duration/t_step)
    cfg, vel, acc, H = generate_whip_motion(start_config, mid_config, end_config, H, t_step)

    for t in range(H+1):
        print(f"{t*t_step}\t{' '.join(map(str, cfg[t,:]))}\t{' '.join(map(str, vel[t,:]))}\t{' '.join(map(str, acc[t,:]))}")


def take_action(held_link, at, keyf, settlef):
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + keyf)
    held_link.location += Vector((at[0], at[1], at[2]))
    held_link.keyframe_insert(data_path="location")
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + settlef)
    held_link.keyframe_insert(data_path="location")

def success_ac(rope, obstacle_x, obstacle_y, obstacle_z, obstacle_size_x, obstacle_size_y):
    min_y = inf
    min_z = inf
    suc = 0
    z_suc = 0
    num = 0
    left_bound = obstacle_y + obstacle_size_y
    right_bound = obstacle_y - obstacle_size_y
    up_bound = obstacle_x + obstacle_size_x
    bottom_bound = obstacle_x - obstacle_size_x
    # bpy.context.view_layer.update()
    print(bpy.context.scene.frame_current)
    for r in rope:
        # if r.matrix_world.translation[1] <= min_y:
        #     min_y = r.matrix_world.translation[1]
        #     if r.matrix_world.translation[1] < right_bound:
        #         suc += 1
        #     if r.matrix_world.translation[1] > left_bound:
        #         left += 1
        bpy.context.view_layer.update()
        # print(r, r.matrix_world.translation, up_bound, bottom_bound, right_bound)
        if r.matrix_world.translation[0] <= up_bound and r.matrix_world.translation[0] >= bottom_bound:
            # print(r.matrix_world.translation[1])
            if r.matrix_world.translation[1] <= right_bound:
                suc += 1
                if r.matrix_world.translation[2] <= obstacle_z:
                    z_suc += 1
        if r.matrix_world.translation[1] <= right_bound:
            num += 1
    # print(suc)
    return suc > 1 and num > 1

def success_knocking(target_object, obstacle_height):
    return target_object.matrix_world.translation[2] < obstacle_height

def success_weaving(rope, obstacle_x_1, obstacle_y_1, obstacle_z_1, obstacle_size_x_1, obstacle_size_y_1, obstacle_x_2, obstacle_y_2, obstacle_z_2, obstacle_size_x_2, obstacle_size_y_2):
    min_y = inf
    min_z = inf
    suc = 0
    z_suc = 0
    num = 0
    left_bound = min(obstacle_y_1 + obstacle_size_y_1, obstacle_y_2 + obstacle_size_y_2)
    right_bound = max(obstacle_y_1 - obstacle_size_y_1, obstacle_y_2 - obstacle_size_y_2)
    up_bound = obstacle_x_2 - obstacle_size_x_2
    bottom_bound = obstacle_x_1 + obstacle_size_x_1
    for r in rope:
        bpy.context.view_layer.update()
        if r.matrix_world.translation[0] <= up_bound and r.matrix_world.translation[0] >= bottom_bound:
            if r.matrix_world.translation[1] <= right_bound:
                suc += 1
                if r.matrix_world.translation[2] <= min(obstacle_z_1, obstacle_z_2):
                    z_suc += 1
    return suc > 1 and z_suc > 1

def create_obstacle(obstacle_height, obstacle_x, obstacle_y, obstacle_loc):
    bpy.ops.mesh.primitive_cube_add(rotation=(0, 0, 0), location=obstacle_loc)
    bpy.ops.rigidbody.object_add()
    bpy.ops.transform.resize(value=(obstacle_x,obstacle_y,obstacle_height/2))
    obstacle = bpy.context.object
    obstacle.rigid_body.type = 'PASSIVE'
    obstacle.rigid_body.friction = 0.7

    if image:
        mat = bpy.data.materials.new(name="white")
        mat.diffuse_color = (1, 1, 1, 0)    
        obstacle.data.materials.append(mat)

    return obstacle

def create_walls():
    bpy.ops.mesh.primitive_cube_add(location=(19.5, 0, 0))
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    bpy.ops.transform.resize(value=(0.5,40,15))
    wall = bpy.context.object
    wall_color = bpy.data.materials.new(name="black")
    wall_color.diffuse_color = (0, 0, 0, 1)    
    wall.data.materials.append(wall_color)

    bpy.ops.mesh.primitive_cube_add(location=(0, -10, 0))
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    bpy.ops.transform.resize(value=(40,0.5,15))
    wall = bpy.context.object 
    wall.data.materials.append(wall_color)

def add_color(rope, color, color_name):
    mat = bpy.data.materials.new(name=color_name)
    mat.diffuse_color = color
    for r in rope:
        r.data.materials.append(mat)

def get_highest_z(rope):
    bpy.context.view_layer.update()
    return np.max([r.matrix_world.translation[2] for r in rope])

def get_capsule_coords(rope):
    bpy.context.view_layer.update()
    return [r.matrix_world.translation[:3] for r in rope]

if "__main__" == __name__:
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-num', '--num_iterations', dest='num_iterations', type=int)
    parser.add_argument('-image', '--image', dest='image', type=int)
    parser.add_argument('-mode', '--mode', dest='mode', type=str)
    parser.add_argument('-task', '--task', dest='task', type=str)
    args = parser.parse_known_args(argv)[0]

    # Set up scene
    clear_scene()
    add_camera_light()
    create_walls()
    rope = make_rope_v3(params)
    add_color(rope, (0.8, 0.085, 0, 1), 'orange')
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

    ur5 = UR5()
    # 1. Scale the UR5 base and move it
    ur5.base.scale[0] = 4
    ur5.base.scale[1] = 4
    ur5.base.scale[2] = 4
    # ur5.base.location[0] = -4
    ur5.base.location[0] = 1
    ur5.base.location[1] = 7.5
    ur5.base.location[2] = 7.5
    # ur5.base.location[2] = 2
    bpy.context.view_layer.objects.active = ur5.base
    bpy.context.object.rotation_mode = 'XYZ'
    ur5.base.rotation_euler = Euler((0, 0, pi/2 + pi), 'XYZ')
    bpy.ops.object.transform_apply(location = False, scale = True, rotation = False)
    bpy.context.view_layer.update()

    N = args.num_iterations
    image = args.image
    mode = args.mode
    task = args.task

    if mode == 'EVAL':
        f = open('eval_settings.txt', 'r')

        obstacle_size_xs = []
        obstacle_size_ys = []
        obstacle_heights = []
        obstacle_locs = []
        apex_points = []

        for setting in f:
            obstacle_size_x, obstacle_size_y, obstacle_height, obstacle_x, obstacle_y, joint_1, joint_2, joint_3 = setting.split(' ')
            obstacle_size_xs.append(float(obstacle_size_x) * 2)
            obstacle_size_ys.append(float(obstacle_size_y) * 2)
            obstacle_heights.append(float(obstacle_height) * 4)
            obstacle_locs.append((float(obstacle_y) * 4 + 1, -1 * float(obstacle_x) * 4, -1 + float(obstacle_height) * 4 / 2))
            apex_points.append([float(joint_1), float(joint_2), float(joint_3)])

        for i, apex_point in enumerate(apex_points):
            bpy.context.scene.frame_set(1)
            bpy.context.view_layer.update()
  
            for ac in bpy.data.actions:
                bpy.data.actions.remove(ac)

            keyf = 20

            obstacle = create_obstacle(obstacle_heights[i], obstacle_size_xs[i], obstacle_size_ys[i], obstacle_locs[i])
            print("Obstacle location: ", obstacle_locs[i])
            print("Obstacle dims: ", obstacle_size_xs[i], obstacle_size_ys[i], obstacle_heights[i])

            if task == 'KNOCKING':
                # obstacle_top_height = 10
                target_object_height = 0.7
                # obstacle_top_loc = (obstacle_locs[i][0], obstacle_locs[i][1], obstacle_heights[i] + obstacle_top_height/2 - 1 + target_object_height)
                # obstacle_top = create_obstacle(obstacle_top_height, 0.1, 0.1, obstacle_top_loc)
                target_object_loc = (obstacle_locs[i][0], obstacle_locs[i][1], obstacle_heights[i] + target_object_height/2 - 1)
                target_object = create_obstacle(target_object_height, 0.2, 0.2, target_object_loc)
                target_object.rigid_body.type = 'ACTIVE'
            elif task == "WEAVING":
                obstacle_2 = create_obstacle(obstacle_heights[i], obstacle_size_xs[i], obstacle_size_ys[i], (17, 0, -1+obstacle_heights[i]/2))
                obstacle_3 = create_obstacle(obstacle_heights[i], obstacle_size_xs[i], obstacle_size_ys[i], (12, 0, -1+obstacle_heights[i]/2))

            bpy.ops.object.select_all(action='DESELECT')

            d2r = pi/180.

            start_config = np.array([-40.18, -27.27, 68.59,  -152.87, -82.32, -144.38 ])*d2r
            end_config   = np.array([-138.9,  -17.8, 72.54, -152.87, -117.25, -213.24])*d2r
            mid_config = np.append(np.array(apex_point)/d2r, np.array([-169.95, -89.07, -189.]))
            print("Apex point: ", apex_point)

            ur5.set_config(start_config)
            ur5.keyframe_insert(1)
            bpy.context.scene.frame_set(1)

            # Move held link to the upper-left to ensure rope is to the left of obstacle
            ur5.base.location[0] = 1
            ur5.base.location[1] = 7.5
            ur5.base.location[2] = 7.5
            bpy.context.view_layer.objects.active = ur5.base
            bpy.ops.object.transform_apply(location = True, scale = False, rotation = False)
            bpy.context.view_layer.update()

            # 2a. A keyframe motion of held link to put in gripper (start kf = 1, end kf = 80)
            held_link.keyframe_insert(data_path="location")
            held_link.keyframe_insert(data_path="rotation_euler")
            bpy.context.scene.rigidbody_world.enabled = True
            bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
            target_loc = (ur5.gripper.right_inner_finger_pad.matrix_world.translation + ur5.gripper.left_inner_finger_pad.matrix_world.translation)/2
            at = target_loc - held_link.matrix_world.translation
            take_action(held_link, at, keyf, 40-keyf)
            ur5.keyframe_insert(keyf)

            # Move held link to gripper
            ur5.base.location[1] = 0
            ur5.base.location[2] = 1.5
            bpy.context.view_layer.objects.active = ur5.base
            bpy.ops.object.transform_apply(location = True, scale = False, rotation = False)
            bpy.context.view_layer.update()

            held_link.keyframe_insert(data_path="location")
            held_link.keyframe_insert(data_path="rotation_euler")
            bpy.context.scene.rigidbody_world.enabled = True
            bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
            target_loc = (ur5.gripper.right_inner_finger_pad.matrix_world.translation + ur5.gripper.left_inner_finger_pad.matrix_world.translation)/2
            at = target_loc - held_link.matrix_world.translation
            take_action(held_link, at, 20, 20)
            ur5.keyframe_insert(60)

            # 2b. Set keyframe to ur5
            ur5.keyframe_insert(101)
            for frame in range(1, 102):
                bpy.context.scene.frame_set(frame)

            # 3. Add rigid-body constraint to attach held end to ur5.gripper.gripper_base
            # Inverse the transform matrix to make the transform correct
            held_link.parent = ur5.gripper.right_inner_finger_pad
            held_link.matrix_parent_inverse = ur5.gripper.right_inner_finger_pad.matrix_world.inverted()

            # 4. make held link.kinematic=False
            held_link.rigid_body.kinematic = True
            # 5. ur5.set_config(...), insert keyframes
            ur5.set_config(start_config)
            ur5.keyframe_insert(101)
            ur5.set_config(mid_config*d2r)
            ur5.keyframe_insert(111)
            ur5.set_config(end_config)
            ur5.keyframe_insert(121)

            if task == 'WEAVING':
                mid_config_2 = np.array([   -75.76,  -32.7 , 22.68, -169.95, -89.07, -189.])

                ur5.set_config(end_config)
                ur5.keyframe_insert(201)
                ur5.set_config(mid_config_2*d2r)
                ur5.keyframe_insert(211)
                ur5.set_config(start_config)
                ur5.keyframe_insert(221)

            for frame in range(101, 300):
                # print(rope[50], rope[50].matrix_world.translation)
                bpy.context.scene.frame_set(frame)
            
            for frame in range(1, 300):
                bpy.context.scene.frame_set(frame)
                # if frame == 1:
                #     obstacle.hide_render = obstacle.hide_viewport = True
                # elif frame == 51:
                #     obstacle.hide_render = obstacle.hide_viewport = False

            if task == 'VAULTING':
                success = success_ac(rope, obstacle_locs[i][0], obstacle_locs[i][1], obstacle_locs[i][2], obstacle_size_xs[i], obstacle_size_ys[i])
            elif task == 'KNOCKING':
                success = success_knocking(target_object, obstacle_heights[i])
            elif task == 'WEAVING':
                success = success_weaving(rope, obstacle_locs[i][0], obstacle_locs[i][1], obstacle_locs[i][2], obstacle_size_xs[i], obstacle_size_ys[i], 12, 0, -1+obstacle_heights[i]/2, obstacle_size_xs[i], obstacle_size_ys[i]) and success_weaving(rope, 12, 0, -1+obstacle_heights[i]/2, obstacle_size_xs[i], obstacle_size_ys[i], 17, 0, -1+obstacle_heights[i]/2, obstacle_size_xs[i], obstacle_size_ys[i])

            print("Success: ", success)

            # Do not delete on last trial to preserve animation
            if i < len(apex_points) - 1:
                # Delete the obstacle
                bpy.ops.object.select_all(action='DESELECT')
                bpy.context.view_layer.objects.active = obstacle
                obstacle.name="obstacle"
                bpy.data.objects['obstacle'].select_set(True)
                bpy.ops.object.delete(use_global=False)

                held_link.parent = None

                if task == 'KNOCKING':
                    bpy.ops.object.select_all(action='DESELECT')
                    bpy.context.view_layer.objects.active = obstacle_top
                    obstacle_top.name="obstacle_top"
                    bpy.data.objects['obstacle_top'].select_set(True)
                    bpy.ops.object.delete(use_global=False)

    elif mode == "DATAGEN":    

        mid_pred = []

        seq_no = 0

        while seq_no < N:
            print('Experiment Number: ', seq_no)
            
            # remove all keyframes
            bpy.context.scene.frame_set(1)
            for ac in bpy.data.actions:
                bpy.data.actions.remove(ac)

            keyf = np.random.randint(10, 20)

            obstacle_height = np.random.uniform(1.2, 6)
            obstacle_size_x = np.random.uniform(0.3, 0.6)
            obstacle_size_y = np.random.uniform(0.3, 0.6)
            print("Obstacle height %03f, Obstacle size x %03f, Obstacle size y %03f" %(obstacle_height, obstacle_size_x, obstacle_size_y))
            obstacle_loc = (np.random.uniform(4, 19), -0.05-np.random.uniform(-1.2, 4), -1+obstacle_height/2)
            print("Obstacle loc: ", obstacle_loc)

            obstacle = create_obstacle(obstacle_height, obstacle_size_x, obstacle_size_y, obstacle_loc)

            if task == 'KNOCKING':
                obstacle_top_height = 10
                target_object_height = 1
                obstacle_top_loc = (obstacle_loc[0], obstacle_loc[1], obstacle_height + obstacle_top_height/2 - 1 + target_object_height)
                create_obstacle(obstacle_top_height, 0.1, 0.1, obstacle_top_loc)                    

            d2r = pi/180.

            start_config = np.array([-40.18, -27.27, 68.59,  -152.87, -82.32, -144.38 ])*d2r
            end_config   = np.array([-138.9,  -17.8, 72.54, -152.87, -117.25, -213.24])*d2r
            mid_config_origin = np.array([   -80.70,  -86.7, 15.68])
            mid_config = np.append(mid_config_origin, np.array([-169.95, -89.07, -189.]))

            duration = 2 # seconds
            fps = 24
            # H = ceil(fps*duration)
            # traj = generate_whip_motion(start_config, end_config, H, 1./fps)
            t_step = 0.032
            H = ceil(duration/t_step)

            success = False
            counter = 0
            while not success:
                counter += 1
                traj = None
                # traj, vel, acc, H = generate_whip_motion(start_config, mid_config, end_config, H, 1./fps)
                # while True:
                #     H = H - 1
                #     traj_new = generate_whip_motion(start_config, mid_config, end_config, H, 1./fps)
                #     if traj_new is None:
                #         break
                #     traj = traj_new
                
                ur5.set_config(start_config)
                ur5.keyframe_insert(1)
                bpy.context.scene.frame_set(1)

                # Move held link to the upper-left to ensure rope is to the left of obstacle
                ur5.base.location[0] = 1
                ur5.base.location[1] = 8
                ur5.base.location[2] = 8
                bpy.context.view_layer.objects.active = ur5.base
                bpy.ops.object.transform_apply(location = True, scale = False, rotation = False)
                bpy.context.view_layer.update()

                # 2a. A keyframe motion of held link to put in gripper (start kf = 1, end kf = 50)
                held_link.keyframe_insert(data_path="location")
                held_link.keyframe_insert(data_path="rotation_euler")
                bpy.context.scene.rigidbody_world.enabled = True
                bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
                target_loc = (ur5.gripper.right_inner_finger_pad.matrix_world.translation + ur5.gripper.left_inner_finger_pad.matrix_world.translation)/2
                at = target_loc - held_link.matrix_world.translation
                take_action(held_link, at, keyf, 40-keyf)
                ur5.keyframe_insert(keyf)

                # Re-attach held link to get rope to the left of obstacle if necessary
                ur5.base.location[1] = 0
                ur5.base.location[2] = 1.5
                bpy.context.view_layer.objects.active = ur5.base
                bpy.ops.object.transform_apply(location = True, scale = True, rotation = False)
                bpy.context.view_layer.update()

                held_link.keyframe_insert(data_path="location")
                held_link.keyframe_insert(data_path="rotation_euler")
                bpy.context.scene.rigidbody_world.enabled = True
                bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
                target_loc = (ur5.gripper.right_inner_finger_pad.matrix_world.translation + ur5.gripper.left_inner_finger_pad.matrix_world.translation)/2
                at = target_loc - held_link.matrix_world.translation
                take_action(held_link, at, 20, 20)
                ur5.keyframe_insert(keyf)

                # 2b. Set keyframe to ur5
                ur5.keyframe_insert(101)
                for i in range(1, 102):
                    bpy.context.scene.frame_set(i)

                # 3. Add rigid-body constraint to attach held end to ur5.gripper.gripper_base
                # Inverse the transform matrix to make the transform correct
                held_link.parent = ur5.gripper.right_inner_finger_pad
                held_link.matrix_parent_inverse = ur5.gripper.right_inner_finger_pad.matrix_world.inverted()

                # 4. make held link.kinematic=False
                held_link.rigid_body.kinematic = True
                # 5. ur5.set_config(...), insert keyframes
                kf = 101
                if traj is not None:
                    for t in range(traj.shape[0]):
                        wp = traj[t,:]
                        ur5.set_config(wp)
                        ur5.keyframe_insert(kf)
                        kf = kf + 1
                else:
                    ur5.set_config(start_config)
                    ur5.keyframe_insert(101)
                    ur5.set_config(mid_config*d2r)
                    ur5.keyframe_insert(111)
                    ur5.set_config(end_config)
                    ur5.keyframe_insert(121)

                for i in range(101, 300):
                    bpy.context.scene.frame_set(i)

                success = success_ac(rope, obstacle_loc[0], obstacle_loc[1], obstacle_loc[2], obstacle_size_x, obstacle_size_y)
                if not success:
                    mid_config = np.append(mid_config_origin + np.array([np.random.uniform(4, 10) * random.choice((-1, 1)),  np.random.uniform(4, 10) * random.choice((-1, 1)), np.random.uniform(4, 8)]), np.array([-17.65, 75.18, 0. ]))
                    for f in range(101, 300):
                        ur5.keyframe_delete(f)

                print("Success: ", success, ", Trial: ", counter)

                if counter > 10 and not success:
                    bpy.context.scene.frame_set(101)
                    held_link.parent = None
                    break
            
            bpy.context.scene.frame_set(101)
            if success:
                mid_pred.append(mid_config[:3])
                seq_no += 1

            if image and success:
                if not os.path.exists("./whip_ur5_sa/images"):
                    os.makedirs('./whip_ur5_sa/images')
                # Get the scene
                scene = bpy.context.scene
                # Set render resolution
                scene.render.resolution_x = 256
                scene.render.resolution_y = 256
                scene.render.resolution_percentage = 100
                save_render_path = os.path.join(os.getcwd(), 'whip_ur5_sa/images')
                bpy.context.scene.render.filepath = os.path.join(save_render_path, 'whip_state_%05d.jpg'%(seq_no))
                bpy.context.scene.camera.location = (5, 0, 60)
                bpy.ops.render.render(write_still = True)

            if N > 1 and seq_no < N:
                # Delete the obstacle
                bpy.ops.object.select_all(action='DESELECT')
                bpy.context.view_layer.objects.active = obstacle
                obstacle.name="obstacle"
                bpy.data.objects['obstacle'].select_set(True)
                bpy.ops.object.delete(use_global=False)

                held_link.parent = None
        if not os.path.exists("./whip_ur5_sa"):
            os.makedirs('./whip_ur5_sa')
        np.save(os.path.join(os.getcwd(), 'whip_ur5_sa/a.npy'), np.array(mid_pred))
    elif mode == 'MODEL_EVAL':
        num_successes = 0
        
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        net50 = DistModel(device, 3)
        state_dict = torch.load('../resnet_ur5_model_original.pth', map_location=device)['model_state_dict']
        net50.load_state_dict(state_dict)
        net50.resnet_mean.to(device)

        for _ in range(N):
            bpy.context.scene.frame_set(1)
            for ac in bpy.data.actions:
                bpy.data.actions.remove(ac)

            keyf = np.random.randint(10, 20)

            obstacle_height = np.random.uniform(1.2, 6)
            obstacle_size_x = np.random.uniform(0.3, 0.6)
            obstacle_size_y = np.random.uniform(0.3, 0.6)
            print("Obstacle height %03f, Obstacle size x %03f, Obstacle size y %03f" %(obstacle_height, obstacle_size_x, obstacle_size_y))
            obstacle_loc = (np.random.uniform(4, 19), -0.05-np.random.uniform(-1.2, 4), -1+obstacle_height/2)
            print("Obstacle loc: ", obstacle_loc)

            obstacle = create_obstacle(obstacle_height, obstacle_size_x, obstacle_size_y, obstacle_loc)

            d2r = pi/180.
            start_config = np.array([-40.18, -27.27, 68.59,  -152.87, -82.32, -144.38 ])*d2r
            end_config   = np.array([-138.9,  -17.8, 72.54, -152.87, -117.25, -213.24])*d2r

            mid_config = np.array([-169.95, -89.07, -189.])
            duration = 2 # seconds
            fps = 24
            # H = ceil(fps*duration)
            # traj = generate_whip_motion(start_config, end_config, H, 1./fps)
            t_step = 0.032
            H = ceil(duration/t_step)

            success = False
            ur5.set_config(start_config)
            ur5.keyframe_insert(1)
            bpy.context.scene.frame_set(1)

            # Move held link to the upper-left to ensure rope is to the left of obstacle
            ur5.base.location[0] = 1
            ur5.base.location[1] = 7.5
            ur5.base.location[2] = 7.5
            bpy.context.view_layer.objects.active = ur5.base
            bpy.ops.object.transform_apply(location = True, scale = False, rotation = False)
            bpy.context.view_layer.update()

            # 2a. A keyframe motion of held link to put in gripper (start kf = 1, end kf = 80)
            held_link.keyframe_insert(data_path="location")
            held_link.keyframe_insert(data_path="rotation_euler")
            bpy.context.scene.rigidbody_world.enabled = True
            bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
            target_loc = (ur5.gripper.right_inner_finger_pad.matrix_world.translation + ur5.gripper.left_inner_finger_pad.matrix_world.translation)/2
            at = target_loc - held_link.matrix_world.translation
            take_action(held_link, at, keyf, 40-keyf)
            ur5.keyframe_insert(keyf)

            # Move held link to gripper
            ur5.base.location[1] = 0
            ur5.base.location[2] = 1.5
            bpy.context.view_layer.objects.active = ur5.base
            bpy.ops.object.transform_apply(location = True, scale = False, rotation = False)
            bpy.context.view_layer.update()

            held_link.keyframe_insert(data_path="location")
            held_link.keyframe_insert(data_path="rotation_euler")
            bpy.context.scene.rigidbody_world.enabled = True
            bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
            target_loc = (ur5.gripper.right_inner_finger_pad.matrix_world.translation + ur5.gripper.left_inner_finger_pad.matrix_world.translation)/2
            at = target_loc - held_link.matrix_world.translation
            take_action(held_link, at, 20, 20)
            ur5.keyframe_insert(60)

            # 2b. Set keyframe to ur5
            ur5.keyframe_insert(101)
            for i in range(1, 102):
                bpy.context.scene.frame_set(i)
                if i == 101:
                    if not os.path.exists("./whip_ur5_sa/tests"):
                        os.makedirs('./whip_ur5_sa/tests')
                    scene = bpy.context.scene
                    scene.render.resolution_x = 256
                    scene.render.resolution_y = 256
                    scene.render.resolution_percentage = 100
                    save_render_path = os.path.join(os.getcwd(), 'whip_ur5_sa/tests')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'whip_test.png')
                    bpy.context.scene.camera.location = (5, 0, 60)
                    bpy.ops.render.render(write_still = True)
            in_image = Image.open(os.path.join(save_render_path, 'whip_test.png')).convert("RGB")
            normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            preprocess = transforms.Compose([transforms.ToTensor(), normalize])
            in_image = preprocess(in_image).unsqueeze(0)

            net50.eval()
            config = net50(in_image.to(device)).sample().detach().numpy()[0]
            mid_config_pred = np.append(config, mid_config)
            print(mid_config_pred)

            # traj, vel, acc, H = generate_whip_motion(start_config, mid_config_pred, end_config, H, 1./fps)
            traj = None
            kf = 101

            ur5.set_config(start_config)
            ur5.keyframe_insert(1)
            bpy.context.scene.frame_set(1)

            # 3. Add rigid-body constraint to attach held end to ur5.gripper.gripper_base
            # Inverse the transform matrix to make the transform correct
            held_link.parent = ur5.gripper.right_inner_finger_pad
            held_link.matrix_parent_inverse = ur5.gripper.right_inner_finger_pad.matrix_world.inverted()

            # 4. make held link.kinematic=False
            held_link.rigid_body.kinematic = True
            # 5. ur5.set_config(...), insert keyframes
            if traj is not None:
                for t in range(traj.shape[0]):
                    wp = traj[t,:]
                    # print(wp)
                    ur5.set_config(wp)
                    ur5.keyframe_insert(kf)
                    kf = kf + 1
            else:
                ur5.set_config(start_config)
                ur5.keyframe_insert(101)
                ur5.set_config(mid_config_pred*d2r)
                ur5.keyframe_insert(111)
                ur5.set_config(end_config)
                ur5.keyframe_insert(121)

            for i in range(101, 300):
                bpy.context.scene.frame_set(i)

            success = success_ac(rope, obstacle_loc[0], obstacle_loc[1], obstacle_loc[2], obstacle_size_x, obstacle_size_y)

            if success:
                num_successes += 1

            if N > 1:
                bpy.ops.object.select_all(action='DESELECT')
                bpy.context.view_layer.objects.active = obstacle
                obstacle.name="obstacle"
                bpy.data.objects['obstacle'].select_set(True)
                bpy.ops.object.delete(use_global=False)

                held_link.parent = None
        print("Number of successes: ", str(num_successes))