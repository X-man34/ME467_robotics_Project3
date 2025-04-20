import mujoco as mj
from pathlib import Path
import numpy as np
import mujoco.viewer
import time
from spatialmath import SE3, SO3
from numpy import deg2rad
home_angles = np.array([0, np.pi/2, 0, 0, 0, 0])


def get_pose(data, body_id):
    #Extract the positions
    rot = data.xmat[body_id].reshape(3,3)
    pos = data.xpos[body_id] * 1000 #Convert to mm


    #Convert to transformation matrix
    transformation = np.eye(4) 
    transformation[:3, :3] = rot  
    transformation[:3, 3] = pos 
    return SE3(transformation)

def get_transformation(joint_angles: np.ndarray)-> SE3:
    """
    Given the joint angles, return the transformation matrix of the end effector.
    Dimensions are mm, angles in radians. The Mujoco XML file is in meters. 
    Args:
        joint_angles (np.ndarray): Joint angles of the robot.Joint 1 in at index 0, joint 2 at 1 etc. 

    Returns:
        SO3: Transformation matrix of the end effector.
    """
    dh_table = [[27.5, np.pi/2, 339, joint_angles[0]],
                [250, 0, 0, joint_angles[1]],
                [70, np.pi/2, 0, joint_angles[2]],
                [0, -np.pi/2, 250, joint_angles[3]],
                [0, np.pi/2, 0, joint_angles[4]],
                [0, 0, 95, joint_angles[5]]
                ]
    final_transformation = SE3()
    for row in dh_table:
        final_transformation *= SE3().Rz(row[3]) * SE3().Tz(row[2]) * SE3().Tx(row[0]) * SE3().Rx(row[1])
    return final_transformation

def transformation_to_project_pose_vector(transformation: SE3) -> np.ndarray:
    """
    Given a transformation matrix, return a vector representing it in the format asked by the project. 
    Args:
        transformation (SE3): Transformation matrix of the end effector.

    Returns:
        np.ndarray: Pose vector of the end effector.
    """
    translation = transformation.t
    rotation = transformation.rpy(order='zyx', unit='deg')
    return np.concatenate((translation, rotation.flatten()))


question_1_angles = np.array([0, deg2rad(90), 0, 0, deg2rad(-90), 0])
transformation = get_transformation(question_1_angles)
print(transformation)
print(transformation_to_project_pose_vector(transformation))


xml_path = Path("CAD") / "robot_model.xml"
model = mj.MjModel.from_xml_path(str(xml_path))
mujoco_model_data = mj.MjData(model)

end_effector_body_name="end-effector"
end_effector_body_id = model.body(name=end_effector_body_name).id

# To not repeat code, if for some reason you don't want to watch the awsome visualization and just want to see the boring plots, it will open the viewer and them immediately close it. 
# I just don't know how else to do it without repeating the for loop outside the with statement. 
with mujoco.viewer.launch_passive(model, mujoco_model_data) as viewer:
    # viewer.cam.azimuth = 180  # Looking along X-axis
    viewer.cam.distance *= 2.0 
    mujoco_model_data.qpos[:len(question_1_angles)] = question_1_angles
    mj.mj_forward(model, mujoco_model_data)
    viewer.sync()
    print(get_pose(mujoco_model_data, end_effector_body_id))
    while True:
        if not viewer.is_running():
            break
        time.sleep(.001)