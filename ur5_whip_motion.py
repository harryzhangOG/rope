import os
import sys
sys.path.append(os.getcwd())
sys.path.append('/Users/harryzhang/Library/Python/3.7/lib/python/site-packages')
from cvxopt import spmatrix, matrix, solvers, printing
import numpy as np
import math
from math import pi, ceil
from ur5_viz import UR5

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

if __name__ == "__main__":
    import bpy
    # Blender always starts with a cube, delete it.
    bpy.ops.object.delete(use_global=False)

    #start_config = [-pi/3., -pi/6., pi/3, -pi/4, pi/4., 0.]
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
    
    ur5 = UR5()
    kf = 1
    for t in range(traj.shape[0]):
        wp = traj[t,:]
        print(wp)
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
    
