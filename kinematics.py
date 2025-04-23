import numpy as np
from spatialmath import SE3, SO3



class Kinematics:
    def __init__(self, home_joint_angles, dh_table):
        """
        Initialize the Kinematics class with initial joint angles and DH parameters.

        The DH table is a 2 dimensional list, each row being for a differnt link, with as many rows are there are links. 
        Each row has 4 columns. the 0th column is True or False, with true meaning its a revolute joint and False a prismatic one. 
        The succeding 3 values are a, alpha, d in the case of a revolute joint or a, alpha, theta in the case of a prismatic one. 
        This code has not been tested on a robot with a prismatic manipulator. 
        Args:
            inital_joint_angles (np.ndarray): Initial joint angles of the robot.
            dh_table (list): Denavit-Hartenberg parameters for the robot.
        """
        self.home_positon = home_joint_angles
        self.dh_table = dh_table
        self.last_joint_angles = self.home_positon
        self.last_pose = self.foreward(self.home_positon, return_intermediate=True)





    def foreward(self, joint_angles: np.ndarray, return_intermediate=False)-> SE3:
        """
        Given the joint angles, return the transformation matrix of the end effector.
        Dimensions are mm, angles in radians. The Mujoco XML file is in meters. 
        Args:
            joint_angles (np.ndarray): Joint angles of the robot.Joint 1 in at index 0, joint 2 at 1 etc. 
            return_intermediate: If true will return a list of all the transformations. Ths list will have an indentiy at the beginning to make calculations smoother. 

        Returns:
            SO3: Transformation matrix of the end effector.
        """
        self.last_joint_angles = joint_angles
        final_transformation = SE3()
        transformations = [final_transformation]
        for i in range(len(self.dh_table)):
            row = self.dh_table[i]
            if row[0]:
                transformation = SE3().Rz(joint_angles[i]) * SE3().Tz(row[3]) * SE3().Tx(row[1]) * SE3().Rx(row[2])
            else:
                transformation = SE3().Rz(row[3]) * SE3().Tz(joint_angles[i]) * SE3().Tx(row[1]) * SE3().Rx(row[2])#untested
            transformations.append(transformation)
            final_transformation *= transformation
        if return_intermediate:
            return transformations
        else:
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
    
    def jacobian(self, **kwargs):
        """
        Computes the jacobian matrix for a given state. 
        If not passed any args it will use self.last_pose which is the last pose seen by foreward or the home position by default. 
        To specify a custom position pass a set of joint anges into "joint_angles"
        You can also compute the jacobian at an intermidiate link by passing a number into link
        Kwargs:
           joint_angles (list):  the pose to evaluate the jacobian at. 
           link:  the intermediate link to comput the jacobian at. Must be less than the length of the DH table
        """
        if "joint_angles" in kwargs:
            transformations = self.foreward(kwargs.get("joint_angles"), return_intermediate=True)
        else:
            transformations = self.last_pose

        link = kwargs.get("link", len(self.dh_table))
        if link > len(self.dh_table):
            raise ValueError("Link number must be less than length of DH table. ")

        j = np.zeros((6, link))

        for i in range(1, link + 1):
            # For each link
            dh_row = self.dh_table[i - 1]
            z_less_one = transformations[i - 1].R @ np.array([0,0,1])

            if dh_row[0]:
                # Revolute
                j_omega = z_less_one
                j_v = np.linalg.cross(z_less_one, transformations[i].t - transformations[i - 1].t)
            else:
                # Prismatic
                j_omega = np.zeros(3)
                j_v = z_less_one

            j[:, i - 1] = np.hstack((j_v, j_omega)).T
        return j