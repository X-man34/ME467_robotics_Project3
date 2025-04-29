import mujoco as mj
from pathlib import Path
import numpy as np
import mujoco.viewer
import time
from numpy import deg2rad
from kinematics.DHKinematics import DHKinematics
from kinematics import mini_bot_geometric_inverse
import os
from concurrent.futures import ProcessPoolExecutor
import traceback

# def check_solution(proposed_solution: np.ndarray, correct_answer: np.ndarray):
    

#     for i in range(len(correct_answer)):
#         if not np.isclose(proposed_solution[i], correct_answer[i], atol=1e-3):
#             return False
    
#     print(f"Inverse: {proposed_solution}")
#     return True
    

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
    
    # Helps with printouts. 
    np.set_printoptions(suppress=True)

    # Load the mujoco model for answer verification. 
    xml_path = Path("CAD") / "robot_model.xml"
    model = mj.MjModel.from_xml_path(str(xml_path))
    mujoco_model_data = mj.MjData(model)


    end_effector_body_name="end-effector"
    end_effector_body_id = model.body(name=end_effector_body_name).id



    # Compute the transformation of a known position for use later. 
    home_pos = mini_bot_kinematics.foreward(home_angles)

    given_transformation = np.array([[.7551, .4013, .5184, 399.1255], 
                                    [.6084, -.7235, -.3262, 171.01526], 
                                    [.2441, .5617, -.7905, 416.0308], 
                                    [0, 0, 0, 1]])
    
    


    # trials = 1000
    # successes = 0
    # for i in range(trials):
    #     if check_random():
    #         successes += 1
    # success_rate = successes / trials
    # print(f"Success rate: {success_rate} for {trials} trials")


    



    trials = 10000
    workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run_trial, range(trials)))  # no lambda

    successes = sum(results)
    success_rate = successes / trials
    print(f"Success rate: {success_rate} for {trials} trials")

    # question_1_angles = np.array([0, deg2rad(90), 0, 0, deg2rad(-90), 0]) 
    # try:
    #     with mujoco.viewer.launch_passive(model, mujoco_model_data) as viewer:
    #         # viewer.cam.azimuth = 180  # Looking along X-axis
    #         viewer.cam.distance *= 2.0 
    #         mujoco_model_data.qpos[:6] = question_1_angles
    #         mj.mj_forward(model, mujoco_model_data)
    #         viewer.sync()
    #         start_time = time.time()
    #         millis = 0
    #         to_display = 0
    #         while True:
    #             if not viewer.is_running():
    #                 break

    #             # if millis % 3000 == 0:
    #             #     # every thee seconds
    #             #     if to_display < len(solutions):
    #             #         mujoco_model_data.qpos[:6] = solutions[to_display]
    #             #         mj.mj_forward(model, mujoco_model_data)
    #             #         to_display += 1
    #             #     if to_display == len(solutions):
    #             #         to_display = 0
    #             # time.sleep(.001)
    #             millis += 1
    #             # mj.mj_step(model, mujoco_model_data)
    #             viewer.sync()
        
        
        
    # except KeyboardInterrupt:
    #     print("Keyboard interrupt received. Closing viewer...")
    #     viewer.close()