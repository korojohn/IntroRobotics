import numpy as np
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- 1. Βοηθητικές Συναρτήσεις Μετασχηματισμών ---
def RotX(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def RotY(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def RotZ(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

def homogeneous(R, p):
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p.flatten()
    return T

# --- 2. Συνάρτηση Σχεδίασης Κουτιού ---
def draw_box(ax, size, translation, rotation=np.eye(3), color='b', alpha=0.1):
    dx, dy, dz = size
    v = np.array([[0,0,0], [dx,0,0], [dx,dy,0], [0,dy,0],
                  [0,0,dz], [dx,0,dz], [dx,dy,dz], [0,dy,dz]])
    v_world = []
    for vertex in v:
        v_w = (rotation @ vertex.T).T + translation
        v_world.append(v_w)

    faces = [
        [v_world[0], v_world[1], v_world[5], v_world[4]], 
        [v_world[1], v_world[2], v_world[6], v_world[5]], 
        [v_world[2], v_world[3], v_world[7], v_world[6]], 
        [v_world[3], v_world[0], v_world[4], v_world[7]], 
        [v_world[0], v_world[1], v_world[2], v_world[3]], 
        [v_world[4], v_world[5], v_world[6], v_world[7]]  
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha))

# --- 3. Κινηματική Τροχιάς (Cubic Spline σε SE(3)) ---
def cubic_spline_se3_full(T_s, T_g, V_s, V_g, T, num_points):
    tau_s = logm(T_s)
    tau_g = logm(T_g)
    c0 = tau_s
    c1 = V_s
    c2 = (3 / T**2) * (tau_g - tau_s) - (2 / T) * V_s - (1 / T) * V_g
    c3 = (-2 / T**3) * (tau_g - tau_s) + (1 / T**2) * (V_s + V_g)

    time_points = np.linspace(0, T, num_points)
    trajectory_matrices = []

    for t in time_points:
        tau_t = c0 + c1 * t + c2 * t**2 + c3 * t**3
        T_t = expm(tau_t)
        trajectory_matrices.append(T_t)

    return trajectory_matrices

# --- 4. Ορισμός Σημείων & Παραμέτρων ---

V_zero = np.zeros((4, 4))

# T_s: Home Pose
T_s = homogeneous(np.eye(3), np.array([0.0, 0.0, 0.4788]))

# T_g: Top Grasp 
T_g = homogeneous(RotX(np.pi), np.array([0.175, 0.025, 0.05]))

# T_h: Placing
R_h = RotY(np.pi/2) @ RotX(np.pi/4)
T_h = homogeneous(R_h, np.array([0.10, 0.125, 0.10]))

# --- 5. Υπολογισμός Τροχιάς ---
traj1 = cubic_spline_se3_full(T_s, T_g, V_zero, V_zero, T=5, num_points=50)
traj2 = cubic_spline_se3_full(T_g, T_h, V_zero, V_zero, T=5, num_points=50)
full_trajectory = traj1 + traj2

# Εξαγωγή των θέσεων (x, y, z) από τους πίνακες 4x4 για το plot
positions = np.array([T[0:3, 3] for T in full_trajectory])

# --- 6. Σχεδίαση Σκηνής ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 1. Κόκκινο Box
draw_box(ax, [0.20, 0.05, 0.20], [0.0, 0.10, 0.0], color='r', alpha=0.1)

# 2. Μπλε Box
draw_box(ax, [0.05, 0.05, 0.05], [0.15, 0.0, 0.0], color='b', alpha=0.1)

# 3. Ρόμβος οπή
h_side = 0.05
draw_box(ax, [h_side, h_side, h_side],
         translation=np.array([0.10 - 0.025, 0.125 - 0.025, 0.10 - 0.025]),
         rotation=RotY(np.deg2rad(45)), color=[0.5, 0.4, 0.4], alpha=0.7)

# 4. Σχεδίαση Τροχιάς
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Trajectory (SE(3))", color="k", linewidth=2)

# Markers για τα βασικά σημεία
ax.scatter(T_s[0,3], T_s[1,3], T_s[2,3], color='g', s=100, label='Start (Home)')
ax.scatter(T_g[0,3], T_g[1,3], T_g[2,3], color='c', s=100, label='Grasp')
ax.scatter(T_h[0,3], T_h[1,3], T_h[2,3], color='r', s=100, label='Hole')

# Ρυθμίσεις Γραφήματος
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 0.5])
ax.set_zlim([0, 0.50])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title("Robot Workspace: Trajectory & Objects")
ax.legend()
ax.view_init(elev=20, azim=-120)

plt.show()