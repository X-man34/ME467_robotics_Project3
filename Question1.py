import mujoco as mj
from pathlib import Path
import numpy as np
import mujoco.viewer
import time
from spatialmath import SE3, SO3
from numpy import deg2rad
from kinematics import Kinematics
home_angles = np.array([0, np.pi/2, 0, 0, 0, 0])


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

if __name__ == "__main__":
    # Define the paramaters for this robot arm. 
    dh_table = [[27.5, np.pi/2, 339],
                [250, 0, 0],
                [70, np.pi/2, 0],
                [0, -np.pi/2, 250],
                [0, np.pi/2, 0],
                [0, 0, 95]
                ]

    # Create a kinematics object with the home angles and the DH table.
    mini_bot_kinematrics = Kinematics(home_angles, dh_table)

    # Load the mujoco model for answer verification. 
    xml_path = Path("CAD") / "robot_model.xml"
    model = mj.MjModel.from_xml_path(str(xml_path))
    mujoco_model_data = mj.MjData(model)

    end_effector_body_name="end-effector"
    end_effector_body_id = model.body(name=end_effector_body_name).id

    # Verify the kinematrics agree with mujoco at the home position.
    mujoco_model_data.qpos[:len(home_angles)] = home_angles
    mj.mj_forward(model, mujoco_model_data)
    mujoco_home_pose = get_pose(mujoco_model_data, end_effector_body_id)
    print("Home position in mujoco:")
    print(mujoco_home_pose)

    print("Home position in kinematics:")
    print(mini_bot_kinematrics.foreward(home_angles))


    # Verify the kinematrics agree with mujoco at the position for question 1.
    question_1_angles = np.array([0, deg2rad(90), 0, 0, deg2rad(-90), 0]) 
    mujoco_model_data.qpos[:len(question_1_angles)] = question_1_angles
    mj.mj_forward(model, mujoco_model_data)
    mujoco_question_one_pose = get_pose(mujoco_model_data, end_effector_body_id)
    print("Question 1 position in mujoco:")
    print(mujoco_question_one_pose)

    print("Quesiton 1 position from kinematics:")
    print(mini_bot_kinematrics.foreward(question_1_angles))
    
    # transformation = mini_bot_kinematrics.foreward(question_1_angles)
    # print(transformation)
    # print(mini_bot_kinematrics.transformation_to_project_pose_vector(transformation))




    # # To not repeat code, if for some reason you don't want to watch the awsome visualization and just want to see the boring plots, it will open the viewer and them immediately close it. 
    # # I just don't know how else to do it without repeating the for loop outside the with statement. 
    # with mujoco.viewer.launch_passive(model, mujoco_model_data) as viewer:
    #     # viewer.cam.azimuth = 180  # Looking along X-axis
    #     viewer.cam.distance *= 2.0 
    #     mujoco_model_data.qpos[:len(question_1_angles)] = question_1_angles
    #     mj.mj_forward(model, mujoco_model_data)
    #     viewer.sync()
    #     print(get_pose(mujoco_model_data, end_effector_body_id))
    #     while True:
    #         if not viewer.is_running():
    #             break
    #         time.sleep(.001)