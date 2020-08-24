#!/usr/bin/env blender --python

import bpy
import sys, getopt, itertools
from math import pi
from mathutils import Matrix, Vector

# Adds a constraint-based child.  In some cases it may be better to
# use `c.parent = p`.  The difference (other than syntax) seems to be
# that `.parent` changes the hierarchical view of objects in blender,
# whereas contraints are only found through navigating the contraint
# panes of blender.
def add_child(parent, child):
    child.constraints.new(type='CHILD_OF')
    child.constraints['Child Of'].target = parent

# Loads a Collada (.dae) mesh and returns an object for that mesh.
# The mesh may optionally be uniformly scaled.
def load_mesh(name, filepath, scale=1):
    # Mesh files can contain multiple objects.  To simplify things, we
    # create an object without geometry (aka an "empty") to be the
    # parent of everything contained in the mesh file.
    #p = bpy.data.objects.new(name, None)
    #bpy.context.collection.objects.link(p)
    bpy.ops.object.empty_add(radius=0.1)
    p = bpy.context.object
    
    p.rotation_mode = 'AXIS_ANGLE'

    bpy.ops.wm.collada_import(filepath=filepath)
    for c in bpy.context.selected_objects:
        c.parent = p

    if scale != 1:
        bpy.ops.transform.resize(value=(scale, scale, scale))
        
    return p

# Class for loading and tracking the Robotiq 2F 85 gripper mesh objects.
class Gripper:
    def __init__(self):
        meshBase = "ur5/robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_85_"
        
        self.gripper_base = load_mesh("gripper_base", meshBase + "base_link.dae", 0.001)

        self.left_outer_knuckle = load_mesh("left_outer_knuckle", meshBase + "outer_knuckle.dae", 0.001)
        self.left_inner_knuckle = load_mesh("left_inner_knuckle", meshBase + "inner_knuckle.dae", 0.001)
        self.left_outer_finger  = load_mesh("left_outer_finger",  meshBase + "outer_finger.dae", 0.001)
        self.left_inner_finger  = load_mesh("left_inner_finger",  meshBase + "inner_finger.dae", 0.001)
        self.left_inner_finger_pad = load_mesh("left_inner_finger_pad", meshBase + "pad.dae", 0.001)
        
        self.right_outer_knuckle = load_mesh("right_outer_knuckle", meshBase + "outer_knuckle.dae", 0.001)
        self.right_inner_knuckle = load_mesh("right_inner_knuckle", meshBase + "inner_knuckle.dae", 0.001)
        self.right_outer_finger  = load_mesh("right_outer_finger",  meshBase + "outer_finger.dae", 0.001)
        self.right_inner_finger  = load_mesh("right_inner_finger",  meshBase + "inner_finger.dae", 0.001)
        self.right_inner_finger_pad = load_mesh("right_inner_finger_pad", meshBase + "pad.dae", 0.001)

        add_child(self.gripper_base, self.left_outer_knuckle)
        add_child(self.gripper_base, self.left_inner_knuckle)
        add_child(self.left_outer_knuckle, self.left_outer_finger)
        add_child(self.left_outer_finger,  self.left_inner_finger)
        add_child(self.left_inner_finger,  self.left_inner_finger_pad)

        add_child(self.gripper_base, self.right_outer_knuckle)
        add_child(self.gripper_base, self.right_inner_knuckle)
        add_child(self.right_outer_knuckle, self.right_outer_finger)
        add_child(self.right_outer_finger,  self.right_inner_finger)
        add_child(self.right_inner_finger,  self.right_inner_finger_pad)

        self.gripper_base.location = (0, 0.08230, 0)
        self.gripper_base.rotation_axis_angle = (-pi/2, 1, 0, 0)
        
        self.left_outer_knuckle.location = (0, -0.0306011, 0.054904)
        self.left_outer_knuckle.rotation_axis_angle = (pi, 0, 0, 1)
        self.left_inner_knuckle.location = (0, -0.0127, 0.06142)
        self.left_inner_knuckle.rotation_axis_angle = (pi, 0, 0, 1)
        self.left_outer_finger.location = (0, 0.0315, -0.0041)
        self.left_inner_finger.location = (0, 0.0061, 0.0471)

        self.right_outer_knuckle.location = (0, 0.0306011, 0.054904)
        self.right_inner_knuckle.location = (0, 0.0127, 0.06142)
        self.right_outer_finger.location = (0, 0.0315, -0.0041)
        self.right_inner_finger.location = (0, 0.0061, 0.0471)
        
# Class for loading and tracking the UR5 robot meshes.  After loading,
# `set_config` can be used to change the robot's transforms, and
# `keyframe_insert` can be used to insert a keyframe for the
# configuration.
class UR5:
    def __init__(self):
        meshBase="ur5/mesh/visual/"
        
        self.base = load_mesh("Base", meshBase + "Base.dae")

        self.shoulder = load_mesh("Shoulder", meshBase + "Shoulder.dae")
        self.upper_arm = load_mesh("UpperArm", meshBase + "UpperArm.dae")
        self.forearm = load_mesh("Forearm", meshBase + "Forearm.dae")
        self.wrist_1 = load_mesh("Wrist1", meshBase + "Wrist1.dae")
        self.wrist_2 = load_mesh("Wrist2", meshBase + "Wrist2.dae")
        self.wrist_3 = load_mesh("Wrist3", meshBase + "Wrist3.dae")
        
        self.gripper = Gripper()
        
        add_child(self.base, self.shoulder)
        add_child(self.shoulder, self.upper_arm)
        add_child(self.upper_arm, self.forearm)
        add_child(self.forearm, self.wrist_1)
        add_child(self.wrist_1, self.wrist_2)
        add_child(self.wrist_2, self.wrist_3)

        add_child(self.wrist_3, self.gripper.gripper_base)

        self.shoulder.location = (0, 0, 0.089159)
        self.upper_arm.location = (0, 0.13585, 0)
        self.forearm.location = (0, -0.1197, 0.42500)
        self.wrist_1.location = (0, 0, 0.39225)
        self.wrist_2.location = (0, 0.09465, 0)
        self.wrist_3.location = (0, 0, 0.09465)

        self.links = [ self.shoulder, self.upper_arm, self.forearm, self.wrist_1, self.wrist_2, self.wrist_3 ]

    def set_config(self, config):
        self.shoulder.rotation_axis_angle = (config[0] + pi, 0, 0, 1)
        self.upper_arm.rotation_axis_angle = (config[1] + pi/2, 0, 1, 0)
        self.forearm.rotation_axis_angle = (config[2], 0, 1, 0)
        self.wrist_1.rotation_axis_angle = (config[3] + pi/2, 0, 1, 0)
        self.wrist_2.rotation_axis_angle = (config[4], 0, 0, 1)
        self.wrist_3.rotation_axis_angle = (config[5], 0, 1, 0)

    def keyframe_insert(self, frame):
        for link in self.links:
            link.keyframe_insert(data_path='rotation_axis_angle', frame=frame)

def load_traj(fileName):
    with open(fileName) as fp:
        # If we just have a sequence of waypoints, the following 1-liner does the job:
        return [[float(s) for s in line.split()] for line in fp if line]
        #
        # However, now we have a file with multiple "data sets" which
        # are separated by empty lines.  We use 'takewhile' to get the
        # list up to the empty line.
        #lines = itertools.takewhile(lambda x: x.strip(), fp)
        #return [[float(s) for s in line.split()] for line in lines]
    
def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('trajectory', type=str)
    parser.add_argument('--frame_step', type=int, default=5)
    args = parser.parse_args(argv)
    
    if args.trajectory is None:
        sys.exit("FILE must be specified.  Use '-- --help' for help.")

    trajs = [load_traj(args.trajectory)]
        
    # Blender always starts with a cube, delete it.
    bpy.ops.object.delete(use_global=False)

    # Move the camera to a reasonable position (though this position
    # may need refinement...)
    camera = bpy.data.objects['Camera']
    camera.location = (0.274, 3.210, 1.544)
    camera.rotation_euler = (1.129, 0.0, 3.110)

    # Change view to use camera (which requires finding which area of
    # the screen is the 3D view, and then changing it)
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces[0].region_3d.view_perspective = 'CAMERA'
    
    # Create the meshes for the ur5
    ur5 = UR5()

    # Contrary to the naming of the next method, this in fact, deselects all.
    bpy.ops.object.select_all()

    # Add in keyframes.  Blender seems to start at keyframe 1, though
    # it also seems capable of having negative keyframes.
    kf = 1
    for traj in trajs:
        for i in range(len(traj)):
            wp = traj[i]
            print(wp[1:7])
            ur5.set_config(wp[1:7])
            # print(wp[0:7])
            # ur5.set_config(wp[1:7])
            ur5.keyframe_insert(kf)
            
            kf = kf + args.frame_step

    # Set the end frame of the animation.
    bpy.context.scene.frame_end = kf - args.frame_step
    
    # After inserting keyframes, set the current frame to the first frame.
    bpy.context.scene.frame_current = 1

    # Turn off the "extras" display (all the coordinate frame lines)
    #bpy.context.space_data.overlay.show_extras = False

if __name__ == "__main__":
    # since we're running in blender, we need to peel off
    # everything before the '--' in order to get arguments to the
    # script (and not arguments to the blender executable)
    main(sys.argv[sys.argv.index('--')+1:])

