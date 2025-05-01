import numpy as np
from numpy import deg2rad
from kinematics import DHKinematics
from kinematics import mini_bot_geometric_inverse
from spatialmath import SO3, SE3



def get_path(start_pose: SE3, goal_pose: SE3, travel_time, time_step)-> list[SE3]:
    def get_path(start_pose: SE3, goal_pose: SE3, travel_time, time_step) -> list[SE3]:
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
    num_poses = (travel_time / time_step) - 1# Subtract one becuase the goal pose will be added on at the end, but we still want to do it in the right amount of time. 
    pose_linear_spacing = path_length / num_poses

    # we also have to deal with rotation. 
    # Find the axis angle representation of the rotation needed between the two poses. 
    needed_rotation = goal_pose.R @ np.array(start_pose.R).T
    angle, axis = SE3().Rt(needed_rotation, np.array([0,0,0])).UnitQuaternion().angvec()
    angle_step = angle / num_poses
    delta_rot = SO3().AngVec(angle_step, axis)

    # Now we can at least define the translation part of the intermediate poses. 
    intermediate_poses = [start_pose]
    distance_traveled = pose_linear_spacing
    while distance_traveled < path_length:
        t = start_pose.t + path_direction * distance_traveled
        R = intermediate_poses[-1].R @ delta_rot.R
        intermediate_poses.append(SE3().Rt(R, t, check=False))
        distance_traveled += pose_linear_spacing
        
    intermediate_poses.append(goal_pose)# The last pose should be really close to this, but just to get rid of rounding errors, add this at the end. 


if __name__ == "__main__":
    # The goal of this is to move a robot through a series of steps using actuators. 
    # Thats really ambitios, will start with just setting positions and ignoring actuators
    # Steps
    # Define a start and end pose and travel time. 
    # For simplicity will try and move endeffector along a straight line between the two poses and not worry if the robot can physically do this. 
    # Come up with a series of poses spaced out along the path. 
    #   Will use travel time and a constant time step, and distance traveled to determine how many and how far apart the intermediate poses are. 
    # Then every time step will use inverse kinematics to determine needed joint angles and magically set them. Later it would be good to resolve a movement into commanded actuator forces, maybe implement a pid loop idk
    
    
    
    # Define the Denavit-Hartenberg (DH) parameters for this robot arm. 
    dh_table = [[True, 27.5, np.pi/2, 339],
                [True, 250, 0, 0],
                [True, 70, np.pi/2, 0],
                [True, 0, -np.pi/2, 250],
                [True, 0, np.pi/2, 0],
                [True, 0, 0, 95]
                ]

    #Define objects
    home_angles = np.array([0, np.pi/2, 0, 0, 0, 0])
    mini_bot_kinematics = DHKinematics(home_angles, dh_table)   

    # Compute some poses, these could be generated in any number of ways, maybe its where the thing we need to pick up is, but for now I'm just using kinematics because its easy. 
    home_pose = mini_bot_kinematics.foreward(home_angles)

    goal_angles = np.array([0, deg2rad(90), 0, 0, deg2rad(-90), 0]) 
    goal_pose = mini_bot_kinematics.foreward(goal_angles)


    # Define some constants. 
    travel_time = 5# seconds
    time_step = .02# seconds. 

    pose_path = get_path(home_pose, goal_pose, travel_time, time_step)







    pass