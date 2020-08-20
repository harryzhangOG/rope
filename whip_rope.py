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
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def load_resnet50(model_path):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU Cuda 0")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    resnet50 = models.resnet50(pretrained=False)
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(nn.Dropout(0.55), nn.Linear(num_ftrs, 3))
    resnet50.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    resnet50.load_state_dict(checkpoint['model_state_dict'])
    return resnet50, device

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


def success_ac(rope, obstacle_x, obstacle_y, obstacle_z, obstacle_radius):
    min_y = inf
    min_z = inf
    suc = 0
    z_suc = 0
    num = 0
    left_bound = obstacle_y + obstacle_radius
    right_bound = obstacle_y - obstacle_radius
    up_bound = obstacle_x + obstacle_radius
    bottom_bound = obstacle_x - obstacle_radius
    for r in rope:
        # if r.matrix_world.translation[1] <= min_y:
        #     min_y = r.matrix_world.translation[1]
        #     if r.matrix_world.translation[1] < right_bound:
        #         suc += 1
        #     if r.matrix_world.translation[1] > left_bound:
        #         left += 1
        if r.matrix_world.translation[0] <= up_bound and r.matrix_world.translation[0] >= bottom_bound:
            if r.matrix_world.translation[1] <= right_bound:
                suc += 1
                if r.matrix_world.translation[2] <= obstacle_z:
                    z_suc += 1
        if r.matrix_world.translation[1] <= right_bound:
            num += 1
    print(suc)
    return suc > 1 and z_suc > 1 and num > 10


if "__main__" == __name__:
    with open("rigidbody_params.json", "r") as f:
        params = json.load(f)
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-num', '--num_iterations', dest='num_iterations', type=int)
    parser.add_argument('-render', '--render', dest='render', type=int)
    parser.add_argument('-image', '--image', dest='image', type=int)
    parser.add_argument('-mode', '--mode', dest='mode', type=str)

    args = parser.parse_known_args(argv)[0]
    # Number of episodes
    N = args.num_iterations
    render = args.render
    image = args.image
    mode = args.mode

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
    if mode == 'DATAGEN':
        for seq_no in range(N):
            print('Experiment Number: ', seq_no)
            # remove all keyframes
            bpy.context.scene.frame_set(1)
            for ac in bpy.data.actions:
                bpy.data.actions.remove(ac)


            # Default height = 2m
            obstacle_height = np.random.uniform(0.5, 4)
            obstacle_radius = np.random.uniform(0.2, 2)
            print("Obstacle height %03f, Obstacle radius %03f" %(obstacle_height, obstacle_radius))
            # obstacle_loc = (np.random.uniform(0, 18), -2-np.random.uniform(-0.5, 3), -1+obstacle_height/2)
            obstacle_loc = (np.random.uniform(13, 20), -2-np.random.uniform(0.5, 3), -1+obstacle_height/2)
            print("Obstacle loc: ", obstacle_loc)
            bpy.ops.mesh.primitive_cylinder_add(radius=obstacle_radius, rotation=(0, 0, 0), location=obstacle_loc)
            bpy.ops.rigidbody.object_add()
            bpy.ops.transform.resize(value=(1,1,obstacle_height/2))

            cylinder = bpy.context.object
            cylinder.rigid_body.type = 'PASSIVE'
            cylinder.rigid_body.friction = 0.7
            if image:
                mat = bpy.data.materials.new(name="red")
                mat.diffuse_color = (1, 0, 0, 0)    
                cylinder.data.materials.append(mat)

            obstacle_x, obstacle_y, obstacle_z = cylinder.matrix_world.translation
            # target_end = (obstacle_x-1, obstacle_y-obstacle_radius-2, obstacle_z+obstacle_height/2)
            # Reduced action reach
            target_end = (0, obstacle_y-obstacle_radius-2, obstacle_z+obstacle_height/2+1)

            held_link.keyframe_insert(data_path="location")
            held_link.keyframe_insert(data_path="rotation_euler")
            bpy.context.scene.rigidbody_world.enabled = True
            bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
            # Randomly perturb
            at = np.array([np.random.uniform(max(0.5, obstacle_x - held_link.matrix_world.translation[0] - 3), min(10, obstacle_x - held_link.matrix_world.translation[0])), 
                        np.random.uniform(obstacle_y + obstacle_radius + 0.5 - held_link.matrix_world.translation[1], 2), 
                        np.random.uniform(0.1, 1)])

            take_action(held_link, at, 10, 40)
            start_x = held_link.matrix_world.translation[0]

            # The actual parameterized action. We use loc=(-0.225, -1.75, 1.75) as the origin of the action, 
            # which we know works for the obstacle (4.75, -2, 0), with radius=0.5 and height=2
            success = False
            # apred_origin = [obstacle_x-3-held_link.matrix_world.translation[0], 
            #         obstacle_y+obstacle_radius-held_link.matrix_world.translation[1], 
            #         obstacle_height/2+obstacle_z+2-held_link.matrix_world.translation[2]]
            apred_origin = [0, max(-5, target_end[1]-held_link.matrix_world.translation[1]), min(2.5, target_end[2]+2-held_link.matrix_world.translation[2])]
            origin_x, origin_y, origin_z = apred_origin[0], apred_origin[1], apred_origin[2]
            apred = apred_origin.copy()
            counter = 0

            while not success:
                print(apred)
                counter += 1
                bpy.context.scene.frame_set(51)
                take_action(held_link, apred, 9, 0)
                # Fix the end point
                at = [start_x - held_link.matrix_world.translation[0], target_end[1] - held_link.matrix_world.translation[1], target_end[2] - held_link.matrix_world.translation[2]]
                take_action(held_link, at, 9, 0)
                for i in range(1, 121):
                    bpy.context.scene.frame_set(i)
                    if render:
                        # Get the scene
                        scene = bpy.context.scene
                        save_render_path = os.path.join(os.getcwd(), 'whip')
                        bpy.context.scene.render.filepath = os.path.join(save_render_path, 'whip_%d_frame_%03d.jpg'%(seq_no, i))
                        bpy.context.scene.camera.location = (5, 0, 60)
                        bpy.ops.render.render(write_still = True)
                success = success_ac(rope, obstacle_x, obstacle_y, obstacle_z, obstacle_radius)
                # Record Images
                bpy.context.scene.frame_set(51)
                if not success:
                    apred = [origin_x, origin_y+np.random.uniform(0.8, 3), origin_z+np.random.uniform(-1, 1)]
                    bpy.context.scene.frame_set(60)
                    held_link.keyframe_delete(data_path='location')
                    bpy.context.scene.frame_set(69)
                    held_link.keyframe_delete(data_path='location')
                print("Success: ", success)
                if counter > 10 and not success:
                    bpy.context.scene.frame_set(51)
                    break

            a.append(apred)
            if image:
                if not os.path.exists("./whip_policy_sa/images"):
                    os.makedirs('./whip_policy_sa/images')
                # Get the scene
                scene = bpy.context.scene
                # Set render resolution
                scene.render.resolution_x = 256
                scene.render.resolution_y = 256
                scene.render.resolution_percentage = 100
                save_render_path = os.path.join(os.getcwd(), 'whip_policy_sa/images')
                bpy.context.scene.render.filepath = os.path.join(save_render_path, 'whip_state_%05d.jpg'%(seq_no))
                bpy.context.scene.camera.location = (5, 0, 60)
                bpy.ops.render.render(write_still = True)
            
            if N > 1:
                # Delete the obstacle
                bpy.ops.object.delete(use_global=False)

        if not os.path.exists("./whip_policy_sa"):
            os.makedirs('./whip_policy_sa')
        np.save(os.path.join(os.getcwd(), 'whip_policy_sa/a.npy'), np.array(a))
    elif mode == 'EVAL':
        # Load ResNet
        net50, device = load_resnet50(os.path.join(os.getcwd(), 'resnet50_model.pth'))
        net50.eval()

        obstacle_height = np.random.uniform(0.5, 4)
        obstacle_radius = np.random.uniform(0.2, 2)
        print("Obstacle height %03f, Obstacle radius %03f" %(obstacle_height, obstacle_radius))
        obstacle_loc = (np.random.uniform(0, 18), -2-np.random.uniform(-0.5, 3), -1+obstacle_height/2)
        print("Obstacle loc: ", obstacle_loc)
        bpy.ops.mesh.primitive_cylinder_add(radius=obstacle_radius, rotation=(0, 0, 0), location=obstacle_loc)
        bpy.ops.rigidbody.object_add()
        bpy.ops.transform.resize(value=(1,1,obstacle_height/2))

        cylinder = bpy.context.object
        cylinder.rigid_body.type = 'PASSIVE'
        cylinder.rigid_body.friction = 0.7
        mat = bpy.data.materials.new(name="red")
        mat.diffuse_color = (1, 0, 0, 0)    
        if image:
            cylinder.data.materials.append(mat)
        obstacle_x, obstacle_y, obstacle_z = cylinder.matrix_world.translation
        target_end = (obstacle_x-1, obstacle_y-obstacle_radius-2, obstacle_z+obstacle_height/2+1)

        held_link.keyframe_insert(data_path="location")
        held_link.keyframe_insert(data_path="rotation_euler")
        bpy.context.scene.rigidbody_world.enabled = True
        bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
        # Randomly perturb
        at = np.array([np.random.uniform(max(0.5, obstacle_x - held_link.matrix_world.translation[0] - 3), min(10, obstacle_x - held_link.matrix_world.translation[0])), 
                    np.random.uniform(obstacle_y + obstacle_radius + 0.5 - held_link.matrix_world.translation[1], 2), 
                    np.random.uniform(0.1, 1)])

        print(at)
        take_action(held_link, at, 10, 40)
        start_x = held_link.matrix_world.translation[0]        
        for i in range(1, 32):
            bpy.context.scene.frame_set(i)
        # Render test image
        if not os.path.exists("./whip_policy_sa/tests"):
            os.makedirs('./whip_policy_sa/tests')
        # Get the scene
        scene = bpy.context.scene
        # Set render resolution
        scene.render.resolution_x = 256
        scene.render.resolution_y = 256
        scene.render.resolution_percentage = 100
        save_render_path = os.path.join(os.getcwd(), 'whip_policy_sa/tests')
        bpy.context.scene.render.filepath = os.path.join(save_render_path, 'whip_test.png')
        bpy.context.scene.camera.location = (5, 0, 60)
        bpy.ops.render.render(write_still = True)
        
        # Pass rendered image to resnet
        in_image = Image.open(os.path.join(save_render_path, 'whip_test.png')).convert("RGB")
        normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        preprocess = transforms.Compose([transforms.ToTensor(), normalize])
        in_image = preprocess(in_image).unsqueeze(0)
                                         
        apred_origin = net50(in_image.to(device)).detach().numpy()[0]

        origin_x, origin_y, origin_z = apred_origin[0], apred_origin[1], apred_origin[2]
        apred = apred_origin.copy()
        counter = 0
        success = False

        while not success:
            counter += 1
            bpy.context.scene.frame_set(51)
            take_action(held_link, apred, 9, 0)
            # Fix the end point
            at = [start_x - held_link.matrix_world.translation[0], target_end[1] - held_link.matrix_world.translation[1], target_end[2] - held_link.matrix_world.translation[2]]
            take_action(held_link, at, 9, 0)
            for i in range(1, 121):
                bpy.context.scene.frame_set(i)
                if render and i >= 31:
                    # Get the scene
                    scene = bpy.context.scene
                    scene.render.resolution_x = 512
                    scene.render.resolution_y = 512
                    scene.render.resolution_percentage = 100
                    save_render_path = os.path.join(os.getcwd(), 'whip/test')
                    bpy.context.scene.render.filepath = os.path.join(save_render_path, 'whip_frame_%03d.jpg'%(i))
                    bpy.context.scene.camera.location = (5, 0, 60)
                    bpy.ops.render.render(write_still = True)
            success = success_ac(rope, obstacle_x, obstacle_y, obstacle_z, obstacle_radius)
            # Record Images
            bpy.context.scene.frame_set(51)
            if not success:
                apred = [origin_x, origin_y+np.random.uniform(0.5, 3), origin_z+np.random.uniform(0.5, 2)]
                bpy.context.scene.frame_set(60)
                held_link.keyframe_delete(data_path='location')
                bpy.context.scene.frame_set(69)
                held_link.keyframe_delete(data_path='location')
            print("Success: ", success)
            if counter > 10 and not success:
                break

