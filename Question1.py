import mujoco as mj
from pathlib import Path
import numpy as np
import mujoco.viewer
import time
from spatialmath import SE3, SO3
from numpy import deg2rad
from kinematics import Kinematics


  
# Import stuff and define some functions. Other code I wrote is imported from python files. 

import mujoco as mj
from pathlib import Path
import numpy as np
import mujoco.viewer
import time
from spatialmath import SE3, SO3
from spatialmath.base import tr2adjoint
from numpy import deg2rad
from kinematics import Kinematics

def get_pose(data, body_id):
    """
    Extracts the position and rotation of a body from the mujoco data structure.
    """
    #Extract the positions
    rot = data.xmat[body_id].reshape(3,3)
    pos = data.xpos[body_id] * 1000 #Convert to mm


    #Convert to transformation matrix
    transformation = np.eye(4) 
    transformation[:3, :3] = rot  
    transformation[:3, 3] = pos 
    return SE3(transformation)

def geometric_inverse(desired_pose: SE3)-> list:
    """
    Given a desired pose, return the joint angles that achieve it.
    Uses hardcoded values and geometric methods to find the joint angles. 
    """
    #get the position of the wrist center. 
    wrist_center = desired_pose.t - 95 * desired_pose.R @ np.array([0, 0, 1])

    d = 27.5
    r = np.sqrt(wrist_center[0]**2 + wrist_center[1]**2) - d# Projected distance from joint 2 to wrist center in x0 y0 plane. 
    s = wrist_center[2] - 339
    theta1 = np.arctan2(wrist_center[1], wrist_center[0])

    # Now we need to find theta 2 and 3. These are independent of theta1
    
    a2 = 250
    a3 = np.sqrt(70**2 + 250**2)
    extra_theta3 = np.pi - np.arctan(250/70)



    #We find theta3 using the law of cosines. 
    theta3 = np.arccos((r**2 + s**2 - a2**2 - a3**2) / (-2 * a2 * a3)) - extra_theta3

    print(theta3)
    theta2a = np.arctan2(s, r) - np.arctan2(a3 * np.sin(theta3), a2 + a3 * np.cos(theta3))
    print(theta2a)
    # theta2b = np.arctan2(s, r) - np.arctan2(a3 * np.sin(theta3b), a2 + a3 * np.cos(theta3b))

    return [theta1, theta2a, theta3, 0, 0, 0]


def geometric_inverse(desired_pose: np.ndarray)-> list:
    """
    Given a desired pose, return the joint angles that achieve it.
    Uses hardcoded values and geometric methods to find the joint angles. 
    """
    #get the position of the wrist center. d
    desired_translation = desired_pose[:3,3]
    desired_rotation = desired_pose[:3, :3]
    wrist_center = desired_translation - 95 * desired_rotation @ np.array([0, 0, 1])

    d = 27.5
    r = np.sqrt(wrist_center[0]**2 + wrist_center[1]**2) - d# Projected distance from joint 2 to wrist center in x0 y0 plane. 
    s = wrist_center[2] - 339
    theta1 = np.arctan2(wrist_center[1], wrist_center[0])

    # Now we need to find theta 2 and 3. These are independent of theta1
    
    a2 = 250
    a3 = np.sqrt(70**2 + 250**2)
    extra_theta3 = np.pi - np.arctan(250/70)



    #We find theta3 using the law of cosines. 
    cosT3 = 1 / (2 * a2 * a3) * (r**2 + s**2 - a2**2 - a3**2)

    theta3a = np.arctan2(np.sqrt(1 - cosT3**2), cosT3)
    theta3b = np.arctan2(-np.sqrt(1 - cosT3**2), cosT3)

    print(theta3a)
    theta2a = np.arctan2(s, r) - np.arctan2(a3 * np.sin(theta3), a2 + a3 * np.cos(theta3))
    print(theta2a)
    # theta2b = np.arctan2(s, r) - np.arctan2(a3 * np.sin(theta3b), a2 + a3 * np.cos(theta3b))

    return [theta1, theta2a, theta3, 0, 0, 0]

    
if __name__ == "__main__":
    
    # Helps with printouts. 
    np.set_printoptions(suppress=True)

    # Load the mujoco model for answer verification. 
    xml_path = Path("CAD") / "robot_model.xml"
    model = mj.MjModel.from_xml_path(str(xml_path))
    mujoco_model_data = mj.MjData(model)


    end_effector_body_name="end-effector"
    end_effector_body_id = model.body(name=end_effector_body_name).id


    # Define the paramaters for this robot arm. 
    dh_table = [[True, 27.5, np.pi/2, 339],
                [True, 250, 0, 0],
                [True, 70, np.pi/2, 0],
                [True, 0, -np.pi/2, 250],
                [True, 0, np.pi/2, 0],
                [True, 0, 0, 95]
                ]

    home_angles = np.array([0, np.pi/2, 0, 0, 0, 0])
    # Create a kinematics object with the home angles and the DH table.
    mini_bot_kinematics = Kinematics(home_angles, dh_table)
    # Compute the transformation of a known position for use later. 
    home_pos = mini_bot_kinematics.foreward(home_angles)

    given_transformation = np.array([[.7551, .4013, .5184, 399.1255], 
                                    [.6084, -.7235, -.3262, 171.01526], 
                                    [.2441, .5617, -.7905, 416.0308], 
                                    [0, 0, 0, 1]])

    given_transformation = mini_bot_kinematics.foreward(joint_angles=home_angles).A
    asdf = geometric_inverse(given_transformation)
    print(asdf)



    # try:
    #     with mujoco.viewer.launch_passive(model, mujoco_model_data) as viewer:
    #         # viewer.cam.azimuth = 180  # Looking along X-axis
    #         viewer.cam.distance *= 2.0 
    #         asdf = geometric_inverse(home_pos)
    #         # mujoco_model_data.qpos[:6] = [0,np.pi/4, 0, 0, 0, 0]
    #         mujoco_model_data.qpos[:6] = asdf
    #         mj.mj_forward(model, mujoco_model_data)
    #         viewer.sync()
    #         while True:
    #             if not viewer.is_running():
    #                 break
    #             time.sleep(.001)
    #             # mj.mj_step(model, mujoco_model_data)
    #             viewer.sync()
        
        
        
    # except KeyboardInterrupt:
    #     print("Keyboard interrupt received. Closing viewer...")
    #     viewer.close()