import numpy as np
from spatialmath import SE3, SO3



class Kinematics:
    def __init__(self, home_joint_angles, dh_table):
        """
        Initialize the Kinematics class with initial joint angles and DH parameters.

        Args:
            inital_joint_angles (np.ndarray): Initial joint angles of the robot.
            dh_table (list): Denavit-Hartenberg parameters for the robot.
        """
        self.home_positon = home_joint_angles
        self.dh_table = dh_table



    def foreward(self, joint_angles: np.ndarray)-> SE3:
        """
        Given the joint angles, return the transformation matrix of the end effector.
        Dimensions are mm, angles in radians. The Mujoco XML file is in meters. 
        Args:
            joint_angles (np.ndarray): Joint angles of the robot.Joint 1 in at index 0, joint 2 at 1 etc. 

        Returns:
            SO3: Transformation matrix of the end effector.
        """

        final_transformation = SE3()
        for i in range(len(self.dh_table)):
            row = self.dh_table[i]
            final_transformation *= SE3().Rz(joint_angles[i]) * SE3().Tz(row[2]) * SE3().Tx(row[0]) * SE3().Rx(row[1])
        return final_transformation
    
    def transformation_to_project_pose_vector(self, transformation: SE3) -> np.ndarray:
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

    def foreward_as_vector(self, joint_angles: np.ndarray)-> np.ndarray:
        """
        Given the joint angles, return the transformation matrix of the end effector as a vector.
        Dimensions are mm, angles in radians. The Mujoco XML file is in meters. 
        Args:
            joint_angles (np.ndarray): Joint angles of the robot.Joint 1 in at index 0, joint 2 at 1 etc. 

        Returns:
            np.ndarray: Pose vector of the end effector.
        """
        transformation = self.foreward(joint_angles)
        return self.transformation_to_project_pose_vector(transformation)