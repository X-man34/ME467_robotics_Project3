import mujoco as mj
from pathlib import Path
import numpy as np
from spatialmath import SE3
import mujoco.viewer
import time
from numpy import deg2rad
from kinematics import DHKinematics
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
    
    # Helps with printouts. 
    np.set_printoptions(suppress=True)

    # Load the mujoco model for answer verification. 
    xml_path = Path("mujoco_files") / "robot_model.xml"
    model = mj.MjModel.from_xml_path(str(xml_path))
    mujoco_model_data = mj.MjData(model)


    end_effector_body_name="end-effector"
    end_effector_body_id = model.body(name=end_effector_body_name).id



    # Compute the transformation of a known position for use later. 
    home_pos = mini_bot_kinematics.foreward(home_angles)

    result = mini_bot_kinematics.inverse(home_pos, np.array([.01, np.pi/2, 0, 0, 0, .01]), convergence_threshold=1e-8, max_iterations=100, gain=.9)
    print(result[1])
    print(result[0])
    print(mini_bot_kinematics.foreward(result[0]))
    print(home_pos)

    solutions = []
    solutions.append(result[0])
    solutions.append(home_angles)

    question_1_angles = np.array([0, deg2rad(90), 0, 0, deg2rad(-90), 0]) 
    try:
        with mujoco.viewer.launch_passive(model, mujoco_model_data) as viewer:
            # viewer.cam.azimuth = 180  # Looking along X-axis
            viewer.cam.distance *= 2.0 
            mujoco_model_data.qpos[:6] = solutions[0]
            mj.mj_forward(model, mujoco_model_data)
            viewer.sync()
            start_time = time.time()
            millis = 0
            to_display = 0
            while True:
                if not viewer.is_running():
                    break

                if millis % 3000 == 0:
                    # every thee seconds
                    if to_display < len(solutions):
                        mujoco_model_data.qpos[:6] = solutions[to_display]
                        mj.mj_forward(model, mujoco_model_data)
                        to_display += 1
                    if to_display == len(solutions):
                        to_display = 0
                time.sleep(.001)
                millis += 1
                # mj.mj_step(model, mujoco_model_data)
                viewer.sync()
        
        
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Closing viewer...")
        viewer.close()

