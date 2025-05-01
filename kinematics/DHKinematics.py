import numpy as np
from spatialmath import SE3



class DHKinematics:
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
            final_transformation *= transformation
            transformations.append(final_transformation.copy())
        if return_intermediate:
            return transformations
        else:
            return final_transformation
    
    def transformation_to_minimal_representation(self, transformation: SE3) -> np.ndarray:
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
        return self.transformation_to_minimal_representation(transformation)
    
    def jacobian(self, joint_angles, link):
        """
        Computes the jacobian matrix for a given state. 
        If not passed any args it will use self.last_pose which is the last pose seen by foreward or the home position by default. 
        To specify a custom position pass a set of joint anges into "joint_angles"
        You can also compute the jacobian at an intermidiate link by passing a number into link
        the prismatic case is untested, but hopefully is simple enough that it works first try. 
        Kwargs:
           joint_angles (list):  the pose to evaluate the jacobian at. 
           link:  the intermediate link to comput the jacobian at. Must be less than the length of the DH table
        """
        transformations = self.foreward(joint_angles, return_intermediate=True)

        if link > len(self.dh_table):
            raise ValueError("Link number must be less than length of DH table. ")
        
        J = np.zeros((6, link))
        o_n = transformations[-1].t  # Position of end-effector

        for i in range(link):
            # For each link
            dh_row = self.dh_table[i]
            T_i = transformations[i]
            R_i = T_i.R
            o_i = T_i.t
            z_i = R_i @ np.array([0,0,1])  # z-axis of frame i
            if dh_row[0]:
                # Revolute
                J_angular = z_i
                J_linear = np.cross(z_i, o_n - o_i)
            else:
                # Prismatic
                j_omega = np.zeros(3)
                j_v = z_i

            J[:3, i] = J_linear
            J[3:, i] = J_angular
        return J

    def inverse(self, desired_pose: SE3, joint_angles_guess, convergence_threshold = .001, gain=1, max_iterations=30)-> list: 
        """
        Performs a numerical search to determine the set of joint angles that most optimally puts the end-effector at the desired pose. 
        implements both the jacobian transpose method. 
        Args:
            desired_pose (SE(3)): the pose we are trying to get to
            convergence_threshold: if specified will set the maximum magnitude of an update before convergence is decided
            max_iterations: if specified is an alternative exit condition to prevent blocking if it cannot converge. 
            gain: a number that controls the step size. 


        """

        def get_x_desired(pose: SE3)->np.ndarray:
            translation = pose.t
            
            angle, rotation = pose.UnitQuaternion().angvec()
            return np.concatenate((translation, rotation.flatten()))

        # Goal 
        x_desired = get_x_desired(desired_pose)

        q = joint_angles_guess

        converging = True
        iters = 0
        while converging:
            foreward = self.foreward(q)
            error = x_desired - get_x_desired(foreward)
            print(get_x_desired(foreward))
            print(get_x_desired(desired_pose))
            # update = gain * ((self.jacobian(joint_angles=q, link=6).T) @ error)
            update = gain * (np.linalg.pinv(self.jacobian(joint_angles=q, link=6)) @ error)

            q = q + update
            iters += 1
            error = np.linalg.norm(error)
            print(f"Error at iteration {iters} is {error}")
            
            if error < convergence_threshold:
                return q, f"Converged in {iters} iterations, final error: {error}"
            if iters > max_iterations:
                return q, f"Was unable to converge, exited after {iters} iterations, final error was: {error}"
