import mujoco as mj
from pathlib import Path
import numpy as np
import mujoco.viewer
import time
from spatialmath import SE3, SO3
from numpy import deg2rad
from kinematics import Kinematics
from spatialmath.base import tr2rpy, rpy2r, r2x, tr2eul, tr2x


  
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


def adjust_angle(angle):
    # Wrap angle to the range [-pi, pi]
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle

def check_solution(proposed_solution: np.ndarray, correct_answer: np.ndarray):
    

    for i in range(len(correct_answer)):
        if not np.isclose(proposed_solution[i], correct_answer[i], atol=1e-3):
            return False
    
    print(f"Inverse: {proposed_solution}")
    return True
    

def filter_unique_solutions_radians(solutions, tolerance=1e-5):
    """
    Filters out non-unique 6-DOF solutions accounting for 2*pi wraps (in radians).

    Args:
        solutions (list of list of floats): List of 6 joint angles (in radians) for each solution.
        tolerance (float): Numerical tolerance for considering two angles equal.

    Returns:
        list of list of floats: Filtered list with only unique solutions.
    """
    unique_solutions = []

    for sol in solutions:
        sol = np.array(sol)
        is_unique = True
        
        for unique_sol in unique_solutions:
            unique_sol = np.array(unique_sol)

            # Normalize differences to [-pi, pi]
            diff = (sol - unique_sol + np.pi) % (2 * np.pi) - np.pi

            if np.all(np.abs(diff) <= tolerance):
                is_unique = False
                break

        if is_unique:
            unique_solutions.append(sol.tolist())

    return unique_solutions


def geometric_inverse(desired_pose: np.ndarray, kinematics: Kinematics)-> list:
    """
    Given a desired pose, return the joint angles that achieve it.
    Uses hardcoded values and geometric methods to find the joint angles. 
    """

    def get_wrist_center(pose: np.ndarray):
        #get the position of the wrist center. d
        desired_translation = pose[:3,3]
        desired_rotation = pose[:3, :3]
        return desired_translation - 95 * desired_rotation @ np.array([0, 0, 1])
    wrist_center = get_wrist_center(desired_pose)
    d = 27.5
    r = np.sqrt(wrist_center[0]**2 + wrist_center[1]**2) - d# Projected distance from joint 2 to wrist center in x0 y0 plane. 
    s = wrist_center[2] - 339
    theta11 = np.arctan2(wrist_center[1], wrist_center[0])
    theta12 = adjust_angle(theta11 + np.pi)
    possible_theta1s = [theta11, theta12]
    # Now we need to find theta 2 and 3. These are independent of theta1
    
    # length of the second link
    a2 = 250
    # length of line from the 3rd joint to the wrist center. 
    a3 = np.sqrt(70**2 + 250**2)
    # angle between second link and a3
    extra_theta3_large = np.pi - np.arctan(250/70)
    # angle between the 70mm and 250mm segments of link 3
    extra_theta3_small = 1.297788


    # C is the angle between a3 and a2
    cosC = 1 / (-2 * a2 * a3) * (r**2 + s**2 - a2**2 - a3**2)

    Ca = np.arctan2(np.sqrt(1 - cosC**2), cosC)
    Cb = np.arctan2(-np.sqrt(1 - cosC**2), cosC)

    theta31 = np.pi - Ca + extra_theta3_small
    theta32 = np.pi - Cb + extra_theta3_small
    theta33 = Ca - extra_theta3_large
    theta34 = Cb - extra_theta3_large
    theta35 = 2 *  np.pi - extra_theta3_large - Ca
    theta36 = 2 *  np.pi - extra_theta3_large - Cb
    possible_theta3s = [theta31, theta32, theta33, theta34, theta35, theta36]

    c = np.sqrt(r**2 + s**2)
    cosB = 1 / (-2 * a2 * c) * (a3**2 - c**2 - a2**2)
    Ba = np.arctan2(np.sqrt(1 - cosB**2), cosB)
    Bb = np.arctan2(-np.sqrt(1 - cosB**2), cosB)
    triangle_angle = np.atan2(s, r)
    theta21 = Ba + triangle_angle
    theta22 = Bb + triangle_angle
    theta23 = Ba - triangle_angle
    theta24 = Bb - triangle_angle
    possible_theta2s = [theta21, theta22, theta23, theta24]


    # List to store all valid solutions
    valid_solutions = set()
    count = 0
    # For each possible theta1 and theta3, find all possible theta2 combinations
    for theta1 in possible_theta1s:
        for theta2 in possible_theta2s:
            for theta3 in possible_theta3s:
                # Test the forward kinematics for the current set of joint angles
                joint_angles = [theta1, theta2, theta3, 0,0,0]
                this_wrist_center = get_wrist_center(kinematics.foreward(joint_angles).A)
                count += 1
                # Check if the wrist center is in the right place. 
                if np.allclose(this_wrist_center, wrist_center, atol=1e-3):
                   valid_solutions.add(tuple(joint_angles))
    # print(f"Found potential {count} solutions. ")
    # print(f"Found {len(valid_solutions)} unique solutions which have the correct wrist center. ")
    
    # for solution in valid_solutions:
    #     if check_solution(solution, actual_angles):
    #         return valid_solutions# then we have at least one good solution
        
    # print("no correct solutions")


    # Now we have determined the first three joint angles and should have a few unique valid solutions. 
    # We can use this information to construct the rotation matrix to the third frame. Then becuase we already know the final rotation matrix, finding the rotation 
    # from 3 to 6 is trivial. 
    # Once that is determined, the joint angles are euler angles of that matrix. 
    valid_solutions = list(valid_solutions)
    for i in range(len(valid_solutions)):
        solution = list(valid_solutions[i])
        r_0_3 = kinematics.foreward(solution, return_intermediate=True)[3].R
        r_3_6 = np.array(r_0_3).T @ desired_pose[:3, :3]
        zyz_angles = r2x(r_3_6, representation='eul')
        solution[3] = zyz_angles[0]
        solution[4] = zyz_angles[1]
        solution[5] = zyz_angles[2]
        valid_solutions[i] = solution

    valid_solutions = filter_unique_solutions_radians(valid_solutions)

    final_solutions = []
    for solution in valid_solutions:
        if np.allclose(desired_pose, kinematics.foreward(solution).A):
            final_solutions.extend(solution)
    return valid_solutions

    
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
    
    def check_random():     
        try:
            actual_angles = np.random.uniform(low=-np.pi, high=np.pi, size=6)
            # actual_angles = [-2.56062141, 2.81606119, 2.4578946, 3.10282986, 3.09258485, -2.8353843 ]
            given_transformation = mini_bot_kinematics.foreward(joint_angles=actual_angles).A
            # print(f"Actual: {actual_angles}")
            solutions = geometric_inverse(given_transformation, mini_bot_kinematics)

            # print(f"Found: {len(solutions)} solutions. ")
            # print("Correct solutions are:")
            desired_pose = mini_bot_kinematics.foreward(actual_angles).A
            correct_solutions = 0
            for solution in solutions:
                if np.allclose(desired_pose, mini_bot_kinematics.foreward(solution).A):
                    # print(f"Inverse: {solution}")
                    correct_solutions += 1
            return len(solutions) == correct_solutions
        except Exception:
            return False


    # trials = 1000
    # successes = 0
    # for i in range(trials):
    #     if check_random():
    #         successes += 1
    # success_rate = successes / trials
    # print(f"Success rate: {success_rate} for {trials} trials")
    question_1_angles = np.array([0, deg2rad(90), 0, 0, deg2rad(-90), 0]) 
    try:
        with mujoco.viewer.launch_passive(model, mujoco_model_data) as viewer:
            # viewer.cam.azimuth = 180  # Looking along X-axis
            viewer.cam.distance *= 2.0 
            mujoco_model_data.qpos[:6] = question_1_angles
            mj.mj_forward(model, mujoco_model_data)
            viewer.sync()
            start_time = time.time()
            millis = 0
            to_display = 0
            while True:
                if not viewer.is_running():
                    break

                # if millis % 3000 == 0:
                #     # every thee seconds
                #     if to_display < len(solutions):
                #         mujoco_model_data.qpos[:6] = solutions[to_display]
                #         mj.mj_forward(model, mujoco_model_data)
                #         to_display += 1
                #     if to_display == len(solutions):
                #         to_display = 0
                # time.sleep(.001)
                millis += 1
                # mj.mj_step(model, mujoco_model_data)
                viewer.sync()
        
        
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Closing viewer...")
        viewer.close()