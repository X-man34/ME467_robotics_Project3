import time
import mujoco as mj
from pathlib import Path
import numpy as np
from spatialmath import SE3
from kinematics import DHKinematics
dh_table = [[True, 27.5, np.pi/2, 339],
            [True, 250, 0, 0],
            [True, 70, np.pi/2, 0],
            [True, 0, -np.pi/2, 250],
            [True, 0, np.pi/2, 0],
            [True, 0, 0, 95]
            ]

home_angles = np.array([0, np.pi/2, 0, 0, 0, 0])
# Create a kinematics object with the home angles and the DH table.
mini_bot_kinematics = DHKinematics(home_angles, dh_table)


if __name__ == "__main__":
    
    np.set_printoptions(suppress=False)

    # Load the mujoco model for answer verification. 
    xml_path = Path("mujoco_files") / "robot_model.xml"
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)


    end_effector_body_name="end-effector"
    end_effector_body_id = model.body(name=end_effector_body_name).id



    # Compute the transformation of a known position for use later. 
    home_pos = mini_bot_kinematics.foreward(home_angles)
    
    # animate_range_of_motion(angle_limits)


    from kinematics.trajectory import *
    # Define pickup and place poses
    pickup_pose = SE3().CopyFrom([
        [1, 0, 0, 340 * np.cos(deg2rad(60))],
        [0, -1, 0, -340 * np.sin(deg2rad(60))],
        [0, 0, -1, 40],
        [0, 0, 0, 1]
    ], check=False)
    place_pose = SE3().CopyFrom([
        [0, 1, 0, -400],
        [0, 0, 1, 0],
        [1, 0, 0, 100],
        [0, 0, 0, 1]
    ], check=False)

    # Create trajectory
    speed = 100
    time_step = 0.02
    pose_path = get_directional_linear_path(home_pos, np.array([0,0,-1]), 400, speed, time_step)
    pose_path.extend(get_linear_path(pose_path[-1], place_pose , speed, time_step))
    # pose_path += get_hold_path(pickup_pose, 0.375, time_step)
    # pose_path += get_directional_linear_path(pickup_pose, np.array([0,0,1]), 1000, speed, time_step)
    end_pose = mini_bot_kinematics.foreward(np.array([np.pi, np.pi/2, 0,0,0,0]))
    # pose_path = get_three_point_arc_path(home_pos,np.array([0, 20, 1000]), end_pose, speed, time_step)

    # Animate
    animate_path(pose_path, mini_bot_kinematics, home_angles, model, data, time_step, prompt_for_enter=True)