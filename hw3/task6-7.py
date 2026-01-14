import numpy as np
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

global_step = 0

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def screw_axis(w, q):
    w = np.array(w)
    v = -np.cross(w, q)
    return np.concatenate((w, v))

def get_twist_mat(S):
    m = np.zeros((4, 4))
    m[:3, :3] = skew(S[:3])
    m[:3, 3] = S[3:]
    return m

def poe_forward_kinematics(screw_axes, joint_angles, M, q_list):
    T = np.eye(4)
    joint_positions = [np.array(q_list[0])]
    for i in range(len(joint_angles)):
        twist = get_twist_mat(screw_axes[:, i])
        T = T @ expm(twist * joint_angles[i])
        if i < len(joint_angles) - 1:
            q_next_homo = np.append(q_list[i+1], 1)
            joint_positions.append((T @ q_next_homo)[:3])
    ee_pose = T @ M
    joint_positions.append(ee_pose[:3, 3])
    return ee_pose, np.array(joint_positions)

def compute_jacobian(screw_axes, joint_angles, M, q_list):
    n = len(joint_angles)
    J = np.zeros((6, n))
    ee_pose_curr, _ = poe_forward_kinematics(screw_axes, joint_angles, M, q_list)
    p_curr = ee_pose_curr[:3, 3]
    R_curr = ee_pose_curr[:3, :3]
    
    delta = 1e-6
    for i in range(n):
        angles_p = joint_angles.copy()
        angles_p[i] += delta
        ee_pose_p, _ = poe_forward_kinematics(screw_axes, angles_p, M, q_list)
        dp = (ee_pose_p[:3, 3] - p_curr) / delta
        R_err = ee_pose_p[:3, :3] @ R_curr.T
        log_R = np.real(logm(R_err))
        dw = np.array([log_R[2,1], log_R[0,2], log_R[1,0]]) / delta
        J[:, i] = np.concatenate((dp, dw))
    return J

def compute_se3_error(T_curr, T_goal):
    e_p = T_goal[:3, 3] - T_curr[:3, 3]
    R_curr = T_curr[:3, :3]
    R_goal = T_goal[:3, :3]
    R_err = R_goal @ R_curr.T
    log_R = np.real(logm(R_err))
    e_w = np.array([log_R[2,1], log_R[0,2], log_R[1,0]])
    return np.concatenate((e_p, e_w))

# --- Παράμετροι Ρομπότ ---
d1, d3, d5, d7 = 0.1695, 0.1155, 0.1278, 0.0660
q_list = [[0,0,0], [0,0,d1], [0,0,d1+d3], [0,0,d1+d3], [0,0,d1+d3+d5], [0,0,d1+d3+d5], [0,0,d1+d3+d5+d7]]
w_list = [[0,0,1], [0,1,0], [0,0,1], [1,0,0], [0,0,1], [1,0,0], [0,0,1]]
S_list = [screw_axis(w_list[i], q_list[i]) for i in range(7)]
screw_axes = np.array(S_list).T
M = np.eye(4)
M[:3, 3] = [0, 0, d1+d3+d5+d7]

# --- Αρχικοποίηση ---
initial_joint_angles = np.deg2rad([0, 30, 0, -45, 0, -45, 0])
joint_angles = initial_joint_angles.copy()

T_start, _ = poe_forward_kinematics(screw_axes, joint_angles, M, q_list)
T_grasp = np.eye(4); T_grasp[:3, 3] = [0.175, 0.025, 0.05]; T_grasp[:3, :3] = T_start[:3, :3]
T_release = np.eye(4); T_release[:3, 3] = [0.1, 0.125, 0.10]; T_release[:3, :3] = T_start[:3, :3]

# --- GUI ---
grasp_button_pressed = False
release_button_pressed = False

def grasp_callback(event): global grasp_button_pressed; grasp_button_pressed = True
def release_callback(event): global release_button_pressed; release_button_pressed = True

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.2)

ax_grasp = plt.axes([0.3, 0.05, 0.15, 0.075])
ax_release = plt.axes([0.55, 0.05, 0.15, 0.075])
btn_grasp = Button(ax_grasp, 'Grasp', color='lightgreen')
btn_release = Button(ax_release, 'Release', color='salmon')
btn_grasp.on_clicked(grasp_callback); btn_release.on_clicked(release_callback)

def update_plot(joint_coords, obj_pos, title):
    ax.clear()
    ax.plot(joint_coords[:, 0], joint_coords[:, 1], joint_coords[:, 2], '-o', color='blue', linewidth=3, label='Robot Arm')
    ax.scatter(T_grasp[0, 3], T_grasp[1, 3], T_grasp[2, 3], color='red', s=50, label='Grasp Point')
    ax.scatter(T_release[0, 3], T_release[1, 3], T_release[2, 3], color='black', s=50, label='Release Point')
    ax.scatter(obj_pos[0], obj_pos[1], obj_pos[2], color='green', s=100, label='Object')
    ax.set_xlim([-0.2, 0.4]); ax.set_ylim([-0.2, 0.4]); ax.set_zlim([0, 0.5])
    ax.set_title(f"{title}")
    ax.legend(loc='upper left')
    plt.draw()
    plt.pause(0.01)

def task_space_controller(T_target, current_obj_pos, label, hold_object=False):
    global joint_angles, global_step
    steps = 50
    K = 0.5
    for i in range(steps):
        global_step += 1
        T_curr, joint_coords = poe_forward_kinematics(screw_axes, joint_angles, M, q_list)
        error_V = compute_se3_error(T_curr, T_target)
        J = compute_jacobian(screw_axes, joint_angles, M, q_list)
        invJ = J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(6))
        dq = invJ @ error_V
        joint_angles += dq * K
        new_obj_pos = T_curr[:3, 3] if hold_object else current_obj_pos
        update_plot(joint_coords, new_obj_pos, label)
    return T_curr[:3, 3]

def run_simulation():
    global joint_angles, grasp_button_pressed, release_button_pressed, global_step
    
    np.set_printoptions(formatter={'float': '{: .8e}'.format})

    obj_pos = task_space_controller(T_grasp, T_grasp[:3, 3], "Moving to Grasp Point")
    
    while not grasp_button_pressed:
        global_step += 1
        _, joint_coords = poe_forward_kinematics(screw_axes, joint_angles, M, q_list)
        update_plot(joint_coords, obj_pos, "WAITING FOR GRASP...")
        plt.pause(0.1)
    
    T_curr, _ = poe_forward_kinematics(screw_axes, joint_angles, M, q_list)
    print(f"Grasped object.")
    print(T_curr)
    print("-" * 30)

    obj_pos = task_space_controller(T_release, obj_pos, "Carrying Object", hold_object=True)
    
    while not release_button_pressed:
        global_step += 1
        _, joint_coords = poe_forward_kinematics(screw_axes, joint_angles, M, q_list)
        update_plot(joint_coords, obj_pos, "WAITING FOR RELEASE...")
        plt.pause(0.1)

    T_curr, _ = poe_forward_kinematics(screw_axes, joint_angles, M, q_list)
    print(f"Released object.")
    print(T_curr)
    print("-" * 30)
        
    angles_start = joint_angles.copy()
    for i in range(51):
        global_step += 1
        alpha = i / 50
        joint_angles = (1-alpha)*angles_start + alpha*initial_joint_angles
        _, joint_coords = poe_forward_kinematics(screw_axes, joint_angles, M, q_list)
        update_plot(joint_coords, obj_pos, "Returning Home...")

run_simulation()
plt.show()