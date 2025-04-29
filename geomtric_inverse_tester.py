import numpy as np
from numpy import deg2rad
from kinematics.DHKinematics import DHKinematics
from kinematics import mini_bot_geometric_inverse
import os
from concurrent.futures import ProcessPoolExecutor
import traceback


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

def check_random():     
        try:
            actual_angles = np.random.uniform(low=-np.pi, high=np.pi, size=6)
            # actual_angles = [-2.56062141, 2.81606119, 2.4578946, 3.10282986, 3.09258485, -2.8353843 ]
            given_transformation = mini_bot_kinematics.foreward(joint_angles=actual_angles).A
            # print(f"Actual: {actual_angles}")
            solutions = mini_bot_geometric_inverse(given_transformation, mini_bot_kinematics)

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
            traceback.print_exc()
            return False
def run_trial(_):
     return check_random()
if __name__ == "__main__":
    trials = 10000
    workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run_trial, range(trials)))  # no lambda

    successes = sum(results)
    success_rate = successes / trials
    print(f"Success rate: {success_rate} for {trials} trials")
