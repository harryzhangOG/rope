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

    def make_capsule_rope(self, params):
        radius = params["segment_radius"]
        #rope_length = radius * params["num_segments"] * 2 * 0.9 # HACKY -- shortening the rope artificially by 10% for now
        rope_length = radius * params["num_segments"]
        num_segments = int(rope_length / radius)
        separation = radius*1.1 # HACKY - artificially increase the separation to avoid link-to-link collision
        link_mass = params["segment_mass"] # TODO: this may need to be scaled
        link_friction = params["segment_friction"]
        twist_stiffness = 20
        twist_damping = 10
        bend_stiffness = 0
        bend_damping = 5
        num_joints = int(radius/separation)*2+1
        bpy.ops.import_mesh.stl(filepath="capsule_12_8_1_2.stl")
        loc0 = (radius*num_segments,0,0)
        link0 = bpy.context.object
        link0.location = loc0
        loc0 = loc0[0]
        link0.name = "Cylinder"
        bpy.ops.transform.resize(value=(radius, radius, radius))
        link0.rotation_euler = (0, pi/2, 0)
        bpy.ops.rigidbody.object_add()
        link0.rigid_body.mass = link_mass
        link0.rigid_body.friction = link_friction
        link0.rigid_body.linear_damping = params["linear_damping"]
        link0.rigid_body.angular_damping = params["angular_damping"] # NOTE: this makes the rope a lot less wiggly
        #link0.rigid_body.collision_shape = 'CAPSULE'
        bpy.context.scene.rigidbody_world.steps_per_second = 120
        bpy.context.scene.rigidbody_world.solver_iterations = 20
        for i in range(num_segments-1):
            bpy.ops.object.duplicate_move(TRANSFORM_OT_translate={"value":(-2*radius, 0, 0)})
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.rigidbody.connect(con_type='POINT', connection_pattern='CHAIN_DISTANCE')
        bpy.ops.object.select_all(action='DESELECT')
        links = [bpy.data.objects['Cylinder.%03d' % (i) if i>0 else "Cylinder"] for i in range(num_segments)]
        return links

    def make_cable_rig(self, params, bezier):
        bpy.ops.object.modifier_add(type='CURVE')
        #bpy.ops.curve.primitive_bezier_circle_add(radius=0.02)
        bpy.ops.curve.primitive_bezier_circle_add(radius=0.018)
        bezier.data.bevel_object = bpy.data.objects["BezierCircle"]
        bpy.context.view_layer.objects.active = bezier
        return bezier
    
    def createNewBone(self, obj, new_bone_name, head, tail):
        bpy.ops.object.editmode_toggle()
        bpy.ops.armature.bone_primitive_add(name=new_bone_name)
        new_edit_bone = obj.data.edit_bones[new_bone_name]
        new_edit_bone.head = head
        new_edit_bone.tail = tail
        bpy.ops.object.editmode_toggle()
        bone = obj.pose.bones[-1]
        constraint = bone.constraints.new('COPY_TRANSFORMS')
        target_obj_name = "Cylinder" if new_bone_name == "Bone.000" else new_bone_name.replace("Bone", "Cylinder")
        constraint.target = bpy.data.objects[target_obj_name]

    def rig_rope(self, params):
        bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
        arm = bpy.context.object
        n = params["num_segments"]
        radius = params["segment_radius"]
        for i in range(n):
            loc = 2*radius*((n-i) - n//2)
            self.createNewBone(arm, "Bone.%03d"%i, (loc,0,0), (loc,0,1))
        bpy.ops.curve.primitive_bezier_curve_add(location=(radius,0,0))
        bezier_scale = n*radius
        bpy.ops.transform.resize(value=(bezier_scale, bezier_scale, bezier_scale))
        bezier = bpy.context.active_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.curve.select_all(action='SELECT')
        bpy.ops.curve.handle_type_set(type='VECTOR')
        bpy.ops.curve.handle_type_set(type='AUTOMATIC')
        # NOTE: it segfaults for num_control_points > 20 for the braided rope!!
        #num_control_points = 20 # Tune this
        num_control_points = 40 # Tune this
        bpy.ops.curve.subdivide(number_cuts=num_control_points-2)
        bpy.ops.object.mode_set(mode='OBJECT')
        bezier_points = bezier.data.splines[0].bezier_points
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.curve.select_all(action='DESELECT')
        for i in range(num_control_points):
            bpy.ops.curve.select_all(action='DESELECT')
            hook = bezier.modifiers.new(name = "Hook.%03d"%i, type = 'HOOK' )
            hook.object = arm
            hook.subtarget = "Bone.%03d"%(n-1-(i*n/num_control_points))
            pt = bpy.data.curves['BezierCurve'].splines[0].bezier_points[i]
            pt.select_control_point = True
            bpy.ops.object.hook_assign(modifier="Hook.%03d"%i)
            pt.select_control_point = False
        bpy.ops.object.mode_set(mode='OBJECT')
        for i in range(n):
            obj_name = "Cylinder.%03d"%i if i else "Cylinder"
            bpy.data.objects[obj_name].hide_set(True)
            bpy.data.objects[obj_name].hide_render = True
        bezier.select_set(False)
        # rope = make_braid_rig(params, bezier)
        rope = self.make_cable_rig(params, bezier)
    
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



