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

class BlenderEnv():
    def __init__(self, params):
        self.radius = params["segment_radius"]
        self.rope_length = self.radius * params["num_segments"] * 2 * 0.9
        self.num_segments = int(self.rope_length / self.radius)
        self.separation = self.radius * 1.1
        self.link_mass = params["segment_mass"]
        self.link_friction = params["segment_friction"]
        # Parameters for how much the rope resists twisting
        self.twist_stiffness = 20
        self.twist_damping = 10

        # Parameters for how much the rope resists bending
        self.bend_stiffness = 0
        self.bend_damping = 5

        self.num_joints = int(self.radius / self.separation) * 2 + 1
        self.loc0 = self.rope_length / 2

    def clear_scene(self):
        '''Clear existing objects in scene'''
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)
        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)
        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
    
    def make_rope(self, filepath):
        """ 
        FILEPATH is the path to the .stl file to the first link
        """
        bpy.ops.import_mesh.stl(filepath=filepath)
        link0 = bpy.context.object
        link0.name = "link_0"
        bpy.ops.transform.resize(value=(self.radius, self.radius, self.radius))
        link0.rotation_euler = (0, pi/2, 0)
        link0.location = (self.loc0, 0, 0)
        bpy.ops.rigidbody.object_add()
        link0.rigid_body.mass = self.link_mass
        link0.rigid_body.collision_shape = 'CAPSULE'

        self.links = [link0]
        for i in range(1, self.num_segments):
            # Copy link0 to create each additional link
            linki = link0.copy()
            linki.data = link0.data.copy()
            linki.name = "link_" + str(i)
            linki.location = (self.loc0 - i * self.separation, 0, 0)
            bpy.context.collection.objects.link(linki)

            self.links.append(linki)
            # Create a GENERIC_SPRING connecting this link to the previous.
            bpy.ops.object.empty_add(type='ARROWS', radius=self.radius * 2, location=(self.loc0 - (i-0.5) * self.separation, 0, 0))
            bpy.ops.rigidbody.constraint_add(type='GENERIC_SPRING')
            joint = bpy.context.object
            joint.name = 'cc_' + str(i-1) + ':' + str(i)
            rbc = joint.rigid_body_constraint
            # connect the two links
            rbc.object1 = self.links[i-1]
            rbc.object2 = self.links[i]
            # disable translation from the joint.  Note: we can consider
            # making a "stretchy" rope by setting
            # limit_lin_x_{lower,upper} to a non-zero range.
            rbc.use_limit_lin_x = True
            rbc.use_limit_lin_y = True
            rbc.use_limit_lin_z = True
            rbc.limit_lin_x_lower = 0
            rbc.limit_lin_x_upper = 0
            rbc.limit_lin_y_lower = 0
            rbc.limit_lin_y_upper = 0
            rbc.limit_lin_z_lower = 0
            rbc.limit_lin_z_upper = 0
            if self.twist_stiffness > 0 or self.twist_damping > 0:
                rbc.use_spring_ang_x = True
                rbc.spring_stiffness_ang_x = self.twist_stiffness
                rbc.spring_damping_ang_x = self.twist_damping
            if self.bend_stiffness > 0 or self.bend_damping > 0:
                rbc.use_spring_ang_y = True
                rbc.use_spring_ang_z = True
                rbc.spring_stiffness_ang_y = self.bend_stiffness
                rbc.spring_stiffness_ang_z = self.bend_stiffness
                rbc.spring_damping_ang_y = self.bend_damping
                rbc.spring_damping_ang_z = self.bend_damping
            
        # After creating the rope, we connect every link to the link 1
        # separated by a joint that has no constraints.  This prevents
        # collision detection between the pairs of rope points.
        for i in range(2, self.num_segments):
            bpy.ops.object.empty_add(type='PLAIN_AXES', radius=self.radius*1.5, location=(self.loc0 - (i-1) * self.separation, 0, 0))
            bpy.ops.rigidbody.constraint_add(type='GENERIC_SPRING')
            joint = bpy.context.object
            joint.name = 'cc_' + str(i-2) + ':' + str(i)
            joint.rigid_body_constraint.object1 = self.links[i-2]
            joint.rigid_body_constraint.object2 = self.links[i]

        # The following parmaeters seem sufficient and fast for using this
        # rope.  steps_per_second can probably be lowered more to gain a
        # little speed.
        bpy.context.scene.rigidbody_world.steps_per_second = 1000
        bpy.context.scene.rigidbody_world.solver_iterations = 100

        return self.links

    def add_camera_light(self):
        bpy.ops.object.light_add(type='SUN', radius=1, location=(0,0,0), rotation=(36*np.pi/180, -65*np.pi/180, 18*np.pi/180))
        #bpy.ops.object.camera_add(location=(1,-26,5), rotation=(0.8*pi/2,0,0))
        #bpy.ops.object.camera_add(location=(0,0,35), rotation=(0,0,0))
        #bpy.ops.object.camera_add(location=(2,0,28), rotation=(0,0,0))
        bpy.ops.object.camera_add(location=(11,-33,7.5), rotation=(radians(80), 0, radians(16.5)))
        bpy.context.scene.camera = bpy.context.object
    
    def make_table(self, params):
        bpy.ops.mesh.primitive_plane_add(size=params["table_size"], location=(0,0,-0.5))
        bpy.ops.rigidbody.object_add()
        self.table = bpy.context.object
        self.table.rigid_body.type = 'PASSIVE'
        self.table.rigid_body.friction = 0.8
        bpy.ops.object.select_all(action='DESELECT')
    def make_trash_can(self):
        # Used to be (-9, -2, 0.5)
        # bpy.ops.mesh.primitive_cylinder_add(location=(-10, -2.7, 0.5))
        bpy.ops.mesh.primitive_cylinder_add(location=(-10.73851165,  -3.46587803, 0.5))
        bpy.ops.rigidbody.object_add()
        self.can = bpy.context.object
        self.can.rigid_body.type = 'PASSIVE'
        self.can.rigid_body.friction = 0.8
        bpy.ops.object.select_all(action='DESELECT')



