import numpy as np
from kinematics import DHKinematics

from spatialmath.base import tr2eul 
from spatialmath import SE3
def filter_unique_solutions(solutions, tolerance=1e-5):
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
            sol = (sol + np.pi) % (2 * np.pi) - np.pi # Make sure all angles are not two times around. like 2pi->0 
            unique_solutions.append(sol.tolist())

    return unique_solutions
def normalize_angles_rad(angles, tol=1e-5):
    """
    Wrap any angle into [0, 2*pi) and then snap
    anything within `tol` radians of 0 or 2*pi back to 0.
    """
    arr = np.mod(angles, 2 * np.pi)            # Wrap to [0, 2*pi)
    mask = (arr <= tol) | (arr >= 2 * np.pi - tol)  # Snap near 0 or 2*pi
    arr[mask] = 0.0
    return arr.tolist()

def mini_bot_geometric_inverse(desired_pose: np.ndarray, kinematics: DHKinematics, unit='rad')-> list:
    """
    Given a desired pose, return the joint angles that achieve it.
    Uses hardcoded values and geometric methods to find the joint angles. 
    This is a closed form solution that returns as many solutions as are found. It has not been tested at singularities. 
    A number of possible solutions are computed, then a numerical search is performed ad only valid solutions are returned. The function is not guaranteed but is extremely likely to return at least one solution, most often two. 
    From testing it has been determined that sometimes valid solutions are returned, none of which match the angles used to create the transformation. This could be a lingering bug with representing 0 as 2pi or something. 
    """

    def get_wrist_center(pose: np.ndarray):
        #get the position of the wrist center. d
        desired_translation = pose[:3,3]
        desired_rotation = pose[:3, :3]
        return desired_translation - 95 * desired_rotation @ np.array([0, 0, 1])

    wrist_center = get_wrist_center(desired_pose)
    theta11 = np.arctan2(wrist_center[1], wrist_center[0])

    theta12 = theta11 + np.pi
    if theta12 > np.pi:
        theta12 -= 2 * np.pi

    # # possible_theta1s = [theta11, theta12]
    possible_theta1s = [theta11, theta12]
    # Now we need to find theta 2 and 3. These are independent of theta1
    d = 27.5
    r = np.sqrt(wrist_center[0]**2 + wrist_center[1]**2) - d# Projected distance from joint 2 to wrist center in x0 y0 plane.
    r_alternative = np.sqrt(wrist_center[0]**2 + wrist_center[1]**2) + d# In the rare event that theta12 is actually the correct solution we have to add d instead of subtracting it.  
    s = wrist_center[2] - 339
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
    if abs(cosC > 1):
        cosC = 1 / (-2 * a2 * a3) * (r_alternative**2 + s**2 - a2**2 - a3**2)

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
    if (abs(cosB) > 1):
        c = np.sqrt(r_alternative**2 + s**2)
        cosB = 1 / (-2 * a2 * c) * (a3**2 - c**2 - a2**2)
        triangle_angle = np.atan2(s, r_alternative)
    else:
        triangle_angle = np.atan2(s, r) 
    Ba = np.arctan2(np.sqrt(1 - cosB**2), cosB)
    Bb = np.arctan2(-np.sqrt(1 - cosB**2), cosB)
    
    theta21 = Ba + triangle_angle
    theta22 = Bb + triangle_angle
    theta23 = np.pi - Ba - triangle_angle
    theta24 = np.pi - Bb - triangle_angle
    possible_theta2s = [theta21, theta22, theta23, theta24]


    # List to store all valid solutions
    valid_solutions = set()
    count = 0
    # For each possible theta1 and theta3, find all possible theta2 combinations
    # foreward kinematics are computed on the fly to reduce unnesacary computations. 
    # there are certainly more optimizations that can be made, its just not worth the time to make them now.  
    for i in range(len(possible_theta1s)):
        row = kinematics.dh_table[0]
        a_0_1 = SE3().Rz(possible_theta1s[i]) * SE3().Tz(row[3]) * SE3().Tx(row[1]) * SE3().Rx(row[2])
        for j in range(len(possible_theta2s)):
            row = kinematics.dh_table[1]
            a_0_2 = a_0_1 @ (SE3().Rz(possible_theta2s[j]) * SE3().Tz(row[3]) * SE3().Tx(row[1]) * SE3().Rx(row[2]))
            for k in range(len(possible_theta3s)):
                row = kinematics.dh_table[2]
                a_0_3 = a_0_2 @ (SE3().Rz(possible_theta3s[k]) * SE3().Tz(row[3]) * SE3().Tx(row[1]) * SE3().Rx(row[2])) @ kinematics.T_4_6_zero_joints
                # Test the forward kinematics for the current set of joint angles
                joint_angles = [possible_theta1s[i], possible_theta2s[j], possible_theta3s[k], 0,0,0]
                this_wrist_center = get_wrist_center(a_0_3.A)
                count += 1
                # Check if the wrist center is in the right place. 
                if np.allclose(this_wrist_center, wrist_center, atol=1e-3):
                   valid_solutions.add(tuple(joint_angles))
                #    print(f"Used theta3{k}")


    # Now we have determined the first three joint angles and should have a few unique valid solutions. 
    # We can use this information to construct the rotation matrix to the third frame. Then becuase we already know the final rotation matrix, finding the rotation 
    # from 3 to 6 is trivial. 
    # Once that is determined, the joint angles are euler angles of that matrix. 
    valid_solutions = list(valid_solutions)
    solutions_with_rotations = [] 
    for i in range(len(valid_solutions)):
        solution = list(valid_solutions[i])
        r_0_3 = kinematics.foreward(solution, return_intermediate=True)[3].R
        r_3_6 = np.array(r_0_3).T @ desired_pose[:3, :3]
        zyz_angles_a = tr2eul(r_3_6)
        solution[3] = zyz_angles_a[0]
        solution[4] = zyz_angles_a[1]
        solution[5] = zyz_angles_a[2]
        solutions_with_rotations.append(solution.copy())
        # Here we need to add another solution becuase there are two ways you can compute the orientation. 
        zyz_angles_b = tr2eul(r_3_6, flip=True)
        solution[3] = zyz_angles_b[0]
        solution[4] = zyz_angles_b[1]
        solution[5] = zyz_angles_b[2]
        solutions_with_rotations.append(solution.copy()) 
    valid_solutions = filter_unique_solutions(solutions_with_rotations)
    # Assuming valid_solutions is a list of joint angles in radians
    # valid_solutions = [
    #     normalize_angles_rad(sol, tol=1e-3)  # Set tolerance to desired level
    #     for sol in valid_solutions
    # ]

    # Check once again that the solutions are valid, now considering the whole transformation
    final_solutions = []
    for solution in valid_solutions:
        if np.allclose(desired_pose, kinematics.foreward(solution).A):
            final_solutions.extend(solution)
    if unit == 'deg':
        # Convert all angles in valid_solutions to degrees
        valid_solutions = [np.degrees(solution).tolist() for solution in valid_solutions]
    return valid_solutions

