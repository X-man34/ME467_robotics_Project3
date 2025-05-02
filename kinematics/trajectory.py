import numpy as np
from numpy import deg2rad
from kinematics import DHKinematics
from kinematics import mini_bot_geometric_inverse
from spatialmath import SE3
import mujoco as mj
from pathlib import Path
import numpy as np
from spatialmath import SE3
import mujoco.viewer
import time
from numpy import deg2rad
from kinematics import DHKinematics
from kinematics import mini_bot_geometric_inverse
from scipy.spatial.transform import Rotation as R, Slerp
from typing import List


def get_arc_path(start_pose: SE3, goal_pose: SE3, center: np.ndarray, speed: float, time_step: float) -> list[SE3]:
    """
    Generates a circular-arc trajectory of poses between a start pose and a goal pose in SE(3) space.
    The arc lies in the plane defined by start, goal, and the center point. Rotations are interpolated
    in SO(3) using Slerp, and translations follow a circular path of constant radius.

    If start, goal, and center are colinear, assumes they form a diameter and constructs an arbitrary plane
    by choosing an orthogonal vector to define the rotation plane.

    Args:
        start_pose (SE3): Starting pose containing rotation and translation.
        goal_pose (SE3): Goal pose containing rotation and translation.
        center (np.ndarray): 3D point (x, y, z) defining the center of the arc.
        travel_time (float): Total time to traverse the arc.
        time_step (float): Time interval between consecutive poses.

    Returns:
        list[SE3]: List of SE3 poses along the arc, including start and goal.
    """
    # Extract translation vectors
    p0 = start_pose.t
    p1 = goal_pose.t

    # Vectors from center to endpoints
    r0 = p0 - center
    r1 = p1 - center
    radius = np.linalg.norm(r0)
    if not np.isclose(radius, np.linalg.norm(r1), atol=1e-6):
        raise ValueError("Start and goal are at different distances from center; cannot form a consistent arc.")

    # Compute normal of the plane
    plane_normal = np.cross(r0, r1)
    norm_normal = np.linalg.norm(plane_normal)

    # Handle colinear case: define diameter plane
    if np.isclose(norm_normal, 0):
        # Start and goal are opposite ends of a diameter -> angle pi
        angle_total = np.pi
        # Choose arbitrary axis orthogonal to r0
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(r0, arbitrary)) / radius > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        plane_normal = np.cross(r0, arbitrary)
        axis = plane_normal / np.linalg.norm(plane_normal)
    else:
        axis = plane_normal / norm_normal
        # Compute total signed angle between r0 and r1
        dot = np.dot(r0, r1) / (radius**2)
        dot = np.clip(dot, -1.0, 1.0)
        angle_total = np.arccos(dot)
        if np.dot(np.cross(r0, r1), axis) < 0:
            angle_total = -angle_total


    # Compute arc length and travel time
    arc_length = abs(angle_total) * radius
    travel_time = arc_length / speed

    # Number of intermediate poses (excluding endpoints)
    num_steps = int(np.floor(travel_time / time_step)) - 1
    if num_steps < 1:
        return [start_pose, goal_pose]

    # Setup rotation interpolation
    rot_start = R.from_matrix(start_pose.R)
    rot_goal = R.from_matrix(goal_pose.R)
    slerp = Slerp([0, 1], R.concatenate([rot_start, rot_goal]))

    path = [start_pose]
    for i in range(1, num_steps + 1):
        frac = i / (num_steps + 1)
        theta = angle_total * frac
        # Rotate r0 about axis by theta
        rot_vec = axis * theta
        R_arc = R.from_rotvec(rot_vec)
        p_arc = center + R_arc.apply(r0)
        # Interpolate orientation
        R_interp = slerp(frac).as_matrix()
        path.append(SE3().Rt(R_interp, p_arc, check=False))

    path.append(goal_pose)
    return path

def get_three_point_arc_path(start_pose: SE3, via_point: np.ndarray, goal_pose: SE3, speed: float, time_step: float) -> List[SE3]:
    """
    Generates a circular-arc trajectory in SE(3) passing through three positions: start, via, and goal.
    Orientations are interpolated simply between start_pose.R and goal_pose.R via SLERP.

    Args:
        start_pose (SE3): Start pose (rotation+translation).
        via_point (np.ndarray): 3D point the arc must pass through.
        goal_pose (SE3): Goal pose (rotation+translation).
        speed (float): Linear speed along the arc (units per sec).
        time_step (float): Interval between consecutive poses (sec).

    Returns:
        List[SE3]: Sequence of SE3 poses along the arc.
    """
    # Extract translations
    p0 = start_pose.t
    p1 = goal_pose.t
    pvia = via_point

    # Vectors for plane
    v0 = pvia - p0
    v1 = p1 - p0

    # Plane normal
    normal = np.cross(v0, v1)
    if np.linalg.norm(normal) < 1e-8:
        raise ValueError("Start, via, and goal are colinear; no unique circle plane.")
    normal /= np.linalg.norm(normal)

    # Basis in plane
    u = v0 / np.linalg.norm(v0)
    v = np.cross(normal, u)

    # Project points to 2D
    def to_plane(p):
        return np.array([np.dot(p - p0, u), np.dot(p - p0, v)])

    a2 = np.array([0.0, 0.0])
    b2 = to_plane(pvia)
    c2 = to_plane(p1)

    # Compute circle center in 2D
    def circle_center(p1, p2, p3):
        mid1 = (p1 + p2) / 2
        dir1 = np.array([-(p2 - p1)[1], (p2 - p1)[0]])
        mid2 = (p2 + p3) / 2
        dir2 = np.array([-(p3 - p2)[1], (p3 - p2)[0]])
        A = np.stack([dir1, -dir2], axis=1)
        t = np.linalg.solve(A, mid2 - mid1)
        return mid1 + t[0] * dir1

    center2 = circle_center(a2, b2, c2)
    radius = np.linalg.norm(a2 - center2)

    # Angles in plane
    def ang(p): return np.arctan2(p[1] - center2[1], p[0] - center2[0])
    theta0 = ang(a2)
    theta_v = ang(b2)
    theta1 = ang(c2)

    # Normalize and ensure passing through via
    def norm_ang(x): return (x + 2*np.pi) % (2*np.pi)
    t0, tv, t1 = norm_ang(theta0), norm_ang(theta_v), norm_ang(theta1)
    if t0 < t1:
        if not (t0 < tv < t1): t1 -= 2*np.pi
    else:
        if not (t1 < tv < t0): t1 += 2*np.pi
    delta = t1 - t0

    # Compute travel time
    arc_len = abs(delta) * radius
    travel_time = arc_len / speed

    # Steps
    num_steps = max(1, int(np.floor(travel_time / time_step)) - 1)

    # Orientation interpolation
    rot_start = R.from_matrix(start_pose.R)
    rot_goal = R.from_matrix(goal_pose.R)
    slerp = Slerp([0, 1], R.concatenate([rot_start, rot_goal]))

    path: List[SE3] = [start_pose]
    for i in range(1, num_steps+1):
        frac = i / (num_steps + 1)
        theta = theta0 + delta * frac
        p2 = center2 + radius * np.array([np.cos(theta), np.sin(theta)])
        p3 = p0 + p2[0]*u + p2[1]*v
        Rn = slerp(frac).as_matrix()
        path.append(SE3().Rt(Rn, p3, check=False))

    path.append(goal_pose)
    return path


def get_hold_path(pose: SE3,
                  hold_time: float,
                  time_step: float) -> list[SE3]:
    """
    Generates a hold trajectory by repeating the same SE3 pose for a specified duration.

    Args:
        pose (SE3): The pose to hold.
        hold_time (float): Total time to hold at the given pose.
        time_step (float): Time interval between consecutive holds.

    Returns:
        list[SE3]: List of identical SE3 poses representing the hold trajectory.
    """
    # Determine number of repeats (including initial)
    num_steps = int(np.ceil(hold_time / time_step))
    if num_steps < 1:
        return []

    # Repeat the same pose
    return [pose for _ in range(num_steps + 1)]

def get_linear_path(start_pose: SE3, goal_pose: SE3, speed, time_step)-> list[SE3]:
    """
    Generates a straight-line trajectory of poses between a start pose and a goal pose in SE(3) space.
    The function computes intermediate poses that interpolate linearly in translation and rotationally 
    in SO(3) space between the start and goal poses. The trajectory is sampled at regular time intervals 
    defined by the `time_step`.
    Args:
        start_pose (SE3): The starting pose in SE(3), containing rotation (R) and translation (t).
        goal_pose (SE3): The goal pose in SE(3), containing rotation (R) and translation (t).
        travel_time (float): The total time to travel from the start pose to the goal pose.
        time_step (float): The time interval between consecutive poses in the trajectory.
    Returns:
        list[SE3]: A list of SE(3) poses representing the trajectory from the start pose to the goal pose.
                    The list includes the start pose, intermediate poses, and the goal pose.
    Notes:
        - The function assumes that the input poses are valid SE(3) transformations.
        - The goal pose is explicitly added at the end of the trajectory to ensure accuracy.
        - Rotations are interpolated using axis-angle representation, and translations are interpolated linearly.
    """
    # Now we need to compute our path. 
    # For now it will be just a straight line between
    path_vector_from_start = goal_pose.t - start_pose.t
    path_length = np.linalg.norm(path_vector_from_start)
    path_direction = path_vector_from_start / path_length
    
    # Figure out how many poses we will need and their spacing. 
    travel_time = path_length / speed
    num_poses = (travel_time / time_step) - 1# Subtract one becuase the goal pose will be added on at the end, but we still want to do it in the right amount of time. 
    pose_linear_spacing = path_length / num_poses

    # we also have to deal with rotation. 

    rot_start = R.from_matrix(start_pose.R)
    rot_goal = R.from_matrix(goal_pose.R)
    slerp = Slerp([0, path_length], R.concatenate([rot_start, rot_goal]))

    # Now we can at least define the translation part of the intermediate poses. 
    intermediate_poses = [start_pose]
    distance_traveled = pose_linear_spacing
    while distance_traveled < path_length:
        t = start_pose.t + path_direction * distance_traveled
        R_interp = slerp(distance_traveled).as_matrix()
        intermediate_poses.append(SE3().Rt(R_interp, t, check=False))
        distance_traveled += pose_linear_spacing
        
    intermediate_poses.append(goal_pose)# The last pose should be really close to this, but just to get rid of rounding errors, add this at the end. 
    return intermediate_poses


def get_directional_linear_path(start_pose: SE3, direction: np.ndarray, distance: float, speed: float, time_step: float) -> list[SE3]:
    """
    Generates a straight-line trajectory from the start_pose in a specified direction for a given distance.
    Utilizes the existing get_linear_path function by computing a goal pose offset along the direction vector.

    Args:
        start_pose (SE3): The starting pose (rotation + translation).
        direction (np.ndarray): 3-element vector indicating desired direction in Cartesian space.
                                Need not be normalized.
        distance (float): The distance to travel along the direction.
        travel_time (float): Total time to traverse the path.
        time_step (float): Time interval between consecutive poses.

    Returns:
        List[SE3]: List of SE3 poses from start to goal along the direction.
    """
    # Normalize direction
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-6:
        raise ValueError("Direction vector must be non-zero.")
    unit_dir = direction / dir_norm

    # Compute goal translation
    start_t = start_pose.t
    goal_t = start_t + unit_dir * distance

    # Create goal pose with same orientation as start
    goal_pose = SE3().Rt(start_pose.R, goal_t, check=False)

    # Generate linear path using existing function
    return get_linear_path(start_pose, goal_pose, speed, time_step)


def get_joint_angles(desired_pose: SE3, kinematics: DHKinematics)-> np.ndarray:
    current_angles = np.array(kinematics.last_joint_angles).copy()
    ikine_results = mini_bot_geometric_inverse(desired_pose.A, kinematics)
    if len(ikine_results) == 0:
        print(f"Oh Crap!! No inverse kinematics returned for pose: {desired_pose}, using home angles. You should look into this. ")
        return kinematics.home_positon
    best_diff = 999999999999999999999999999999# we are trying to minimize this. 
    best_solution = None
    for solution in ikine_results:
        diff = np.linalg.norm(current_angles - np.array(solution))
        if diff < best_diff:
            best_diff = diff
            best_solution = solution
    kinematics.last_joint_angles = best_solution
    return best_solution

def animate_path(pose_path: list[SE3], kinematics: DHKinematics, start_angles, model, data, time_step, prompt_for_enter=False):
    kinematics.last_joint_angles = start_angles
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # viewer.cam.azimuth = 180  # Looking along X-axis
            viewer.cam.distance *= 2.0 
            data.qpos[:6] = get_joint_angles(pose_path[0], kinematics)
            mj.mj_forward(model, data)
            viewer.sync()
            if prompt_for_enter:
                input("Press enter to start animation. ")
            start_time = time.time()
            time_simulated = 0
            curr_pose = 1
            while True:
                # Calculate the inverse kinematics and update the model to the new position. 
                if curr_pose < len(pose_path):
                    data.qpos[:6] = get_joint_angles(pose_path[curr_pose], kinematics)
                    curr_pose += 1
                else:
                    break
                time_simulated += time_step
                mujoco.mj_forward(model, data)# This is called pre sleep so we use part of our time step to update the viewer, but this wont be been unil viewer.synyc() is called.
                
                if not viewer.is_running():
                    print("Viewer is closed. Exiting...")
                    break
                    
                # Calculate the time to sleep
                elasped_time = time.time() - start_time
                sleep_time = time_simulated - elasped_time# if this is negative it means that the calculations are taking longer than the time step they are simulating so the simulation will be delayed. 
                if sleep_time > 0:
                    time.sleep(sleep_time)# Sleep enough such that the real time elapsed matches the simlated time elapsed. 
                    
                else:
                    print(f"Warning, simulation is delayed: {-sleep_time * 1000:.2f} ms")

                viewer.sync()
            print("Animation finished or viewer closed. ")
            while True:
                time.sleep(.001)
                # If the animation finished, the viewer will remain open until the user closes it, if the viewer closed the viewer, this will not wait. 
                if not viewer.is_running():
                    print("Viewer is closed. Exiting...")
                    break
    
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Closing viewer...")
        viewer.close()
