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

def toggle_animation(obj, frame, animate):
    # Sets the obj to be animable or non-animable at particular frame
    obj.rigid_body.kinematic = animate
    obj.keyframe_insert(data_path="rigid_body.kinematic", frame=frame)

def random_perturb(pert1, pert2):
    p1_link = rope[pert1]
    p2_link = rope[pert2]
    # p1_link.rigid_body.kinematic = True
    # p2_link.rigid_body.kinematic = True
    toggle_animation(p1_link, 1, True)
    toggle_animation(p2_link, 1, True)
    dz = 0
    dx1 = np.random.uniform(0.5, 2) * random.choice((-1, 1))
    dy1 = np.random.uniform(0.8, 2) * random.choice((-1, 1))

    dx2 = np.random.uniform(0.5, 2) * random.choice((-1, 1))
    dy2 = np.random.uniform(0.8, 2) * random.choice((-1, 1))

    take_action(p1_link, (dx2, dy2, dz), 5, 2)
    take_action(p2_link, (dx2, dy2, dz), 5, 2)
    # p1_link.rigid_body.kinematic = False
    # p2_link.rigid_body.kinematic = False
    toggle_animation(p1_link, 6, False)
    toggle_animation(p2_link, 6, False)


def success_ac(rope, obstacle_y, obstacle_radius):
    min_y = inf
    min_z = inf
    suc = 0
    left = 0
    left_bound = obstacle_y + obstacle_radius
    right_bound = obstacle_y
    for r in rope:
        if r.matrix_world.translation[1] <= min_y:
            min_y = r.matrix_world.translation[1]
            if r.matrix_world.translation[1] < right_bound:
                suc += 1
            if r.matrix_world.translation[1] > left_bound:
                left += 1
    print(suc)
    print(left)
    return suc >= 10 and right_bound >= min_y and left <=20


if "__main__" == __name__:
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-num', '--num_iterations', dest='num_iterations', type=int)
    parser.add_argument('-render', '--render', dest='render', type=int)

    args = parser.parse_known_args(argv)[0]
    # Number of episodes
    N = args.num_iterations
    render = args.render

    clear_scene()
    add_camera_light()
    rope = make_rope_v3(params)
    make_table(params)
    # bpy.context.scene.gravity *= 10
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

    s = []
    a = []
    for seq_no in range(N):
        print('Experiment Number: ', seq_no)
        # remove all keyframes
        bpy.context.scene.frame_set(1)
        for ac in bpy.data.actions:
            bpy.data.actions.remove(ac)

        # Perturb the rope
        pert1 = random.sample(range(1, len(rope)-1), 1)[0] 
        pert2 = random.sample(range(1, len(rope)-1), 1)[0] 
        while pert1 == pert2:
            pert1 = random.sample(range(1, len(rope)-1), 1)[0] 

        # Default height = 2m
        obstacle_height = np.random.uniform(0.5, 4)
        obstacle_radius = np.random.uniform(0.2, 2)
        print("Obstacle height %03f, Obstacle radius %03f" %(obstacle_height, obstacle_radius))
        obstacle_loc = (4.75+np.random.uniform(0.5, 6)*random.choice((-1, 1)), -2-np.random.uniform(-0.5, 3), -1+obstacle_height/2)
        print("Obstacle loc: ", obstacle_loc)
        bpy.ops.mesh.primitive_cylinder_add(radius=obstacle_radius, rotation=(0, 0, 0), location=obstacle_loc)
        bpy.ops.rigidbody.object_add()
        bpy.ops.transform.resize(value=(1,1,obstacle_height/2))

        cylinder = bpy.context.object
        cylinder.rigid_body.type = 'PASSIVE'
        cylinder.rigid_body.friction = 0.7
        if render:
            mat = bpy.data.materials.new(name="red")
            mat.diffuse_color = (1, 0, 0, 0)    
            cylinder.data.materials.append(mat)

        obstacle_x, obstacle_y, obstacle_z = cylinder.matrix_world.translation
        # target_end = (obstacle_x-1, obstacle_y-obstacle_radius-2, obstacle_z+obstacle_height/2)
        # Reduced action reach
        target_end = (obstacle_x-1, obstacle_y-obstacle_radius-1, obstacle_z+obstacle_height/2)

        held_link.keyframe_insert(data_path="location")
        held_link.keyframe_insert(data_path="rotation_euler")
        bpy.context.scene.rigidbody_world.enabled = True
        bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
        # Randomly perturb
        at = np.array([np.random.uniform(2, 15), np.random.uniform(2, 5) * random.choice((-1, 1)), np.random.uniform(0.1, 1)])
        print(at)
        take_action(held_link, at, 10, 20)

        # The actual parameterized action. We use loc=(-0.225, -1.75, 1.75) as the origin of the action, 
        # which we know works for the obstacle (4.75, -2, 0), with radius=0.5 and height=2
        success = False
        # apred_origin = [obstacle_x-3-held_link.matrix_world.translation[0], 
        #         obstacle_y+obstacle_radius-held_link.matrix_world.translation[1], 
        #         obstacle_height/2+obstacle_z+2-held_link.matrix_world.translation[2]]
        apred_origin = [2, target_end[1]-held_link.matrix_world.translation[1], target_end[2]+2-held_link.matrix_world.translation[2]]
        origin_x, origin_y, origin_z = apred_origin[0], apred_origin[1], apred_origin[2]
        apred = apred_origin.copy()
        counter = 0

        while not success:
            counter += 1
            bpy.context.scene.frame_set(31)
            take_action(held_link, apred, 5, 0)
            # Fix the end point
            at = [target_end[0] - held_link.matrix_world.translation[0], target_end[1] - held_link.matrix_world.translation[1], target_end[2] - held_link.matrix_world.translation[2]]
            take_action(held_link, at, 5, 0)
            for i in range(1, 101):
                bpy.context.scene.frame_set(i)
                if render:
                    save_render_path = os.path.join(os.getcwd(), 'whip')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'whip_%d_frame_%03d.jpg'%(seq_no, i))
                    bpy.context.scene.camera.location = (5, 0, 60)
                    bpy.ops.render.render(write_still = True)
            success = success_ac(rope, obstacle_y, obstacle_radius)
            if not success:
                apred = [origin_x+np.random.uniform(0.5, 1)*random.choice((-1, 1)), origin_y+np.random.uniform(0.5, 1), origin_z+np.random.uniform(0.2, 2)]
                bpy.context.scene.frame_set(36)
                held_link.keyframe_delete(data_path='location')
                bpy.context.scene.frame_set(41)
                held_link.keyframe_delete(data_path='location')
            print("Success: ", success)
            if counter > 10 and not success:
                break

        a.append(apred)
        s.append([obstacle_x, obstacle_y, obstacle_height, obstacle_radius])
        
        if N > 1:
            # Delete the obstacle
            bpy.ops.object.delete(use_global=False)

    if not os.path.exists("./whip_policy_sa"):
        os.makedirs('./whip_policy_sa')
    np.save(os.path.join(os.getcwd(), 'whip_policy_sa/s.npy'), np.array(s))
    np.save(os.path.join(os.getcwd(), 'whip_policy_sa/a.npy'), np.array(a))
