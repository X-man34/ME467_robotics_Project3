from spatialmath.base import tr2angvec, qconj
import numpy as np
import pandas as pd
import time
import mujoco as mj
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from filters import Estimator
from spatialmath import SO3
from pathlib import Path
import spatialmath as sm    
from filters import magnetic_north_normalized, TRIAD

def get_quat_from_vec(v_spatial, negate_z=False)-> np.ndarray:
    """
    Generate a quaternion that rotates the Z-axis to align with a given 3D vector.

    Parameters
    ----------
    v_spatial : np.ndarray
        A 3D vector to align the Z-axis with.
    negate_z : bool, optional
        If True, aligns with the negative Z-axis instead of positive (default is False).

    Returns
    -------
    np.ndarray
        Quaternion in [w, x, y, z] format that rotates Z-axis to the given vector.

    Notes
    -----
    - Handles edge cases when vectors are aligned or opposite.
    - MuJoCo expects quaternion format [w, x, y, z].
    """
    # Normalize
    v_spatial = v_spatial / np.linalg.norm(v_spatial)

    # Create a quaternion that rotates z-axis to v_spatial
    if negate_z:
        z_axis = np.array([0, 0, -1])# z is negated because thats how we are defining it. 
    else:
        z_axis = np.array([0, 0, 1])
    
    rot_axis = np.cross(z_axis, v_spatial)
    angle = np.arccos(np.clip(np.dot(z_axis, v_spatial), -1.0, 1.0))

    if np.linalg.norm(rot_axis) < 1e-6:
        # Handle edge cases: already aligned or opposite
        if np.dot(z_axis, v_spatial) > 0:
            quat_v = np.array([1, 0, 0, 0])  # identity
        else:
            quat_v = np.array([0, 1, 0, 0])  # 180 deg around X (arbitrary orthogonal axis)
    else:
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        quat_v = R.from_rotvec(angle * rot_axis).as_quat()  # [x, y, z, w]

    # MuJoCo wants [w, x, y, z]
    return np.roll(quat_v, 1)



def simulate_and_visualize_data(csv_data: pd.DataFrame, time_step: float, estimator: Estimator, do_3D_vis=True, show_extra_vectors = False, show_spatial_coords=False, show_body_coords=False, title="Filter"):
    """
    Simulates sensor data using an estimator filter and visualizes orientation in MuJoCo.

    Parameters
    ----------
    csv_data : pd.DataFrame
        Sensor data with columns ['t', 'mx', 'my', 'mz', 'gyrox', 'gyroy', 'gyroz', 'ax', 'ay', 'az'].
    time_step : float
        Time step (seconds) between each data sample.
    estimator : Estimator
        Estimator object with time_step(), get_estimated_error, get_bias, get_v_hat_a, get_v_hat_m.
    do_3D_vis : bool, optional
        If True, enables 3D MuJoCo visualization (default is True).
    show_extra_vectors : bool, optional
        If True, shows v_hat_a and v_hat_m in the viewer (default is False).
    show_spatial_coords : bool, optional
        If True, shows global frame axes (default is False).
    show_body_coords : bool, optional
        If True, shows body frame axes (default is False).

    Returns
    -------
    tuple
        (times, rotation_angles, bias_estimates, error_estimates, roll, pitch, yaw)

    Notes
    -----
    - Uses Mahony filter or similar via Estimator class.
    - Visualization is real-time and syncs simulated time with wall clock.
    - Handles filter initialization and orientation updating.
    - Optionally visualizes magnetic and accelerometer estimated directions.
    """ 
    #Grab the initial values
    row = csv_data.iloc[0]
    raw_mag_vector = np.array([row['mx'], row['my'], row['mz']])
    raw_accel_vector = np.array([row['ax'], row['ay'], row['az']])
    
    #Initialize variables
    times = []
    rotation_angles = []
    roll = []
    pitch = []
    yaw = []
    bias_estimates = []
    error_estimates = []
    # Initialize the filter. This is where you change the gains. You don't have to pass in initial conditions, but it improves the estimate. 
    # You can also ask it to use the TRIAD initial pose estimatior, but at the time of writing the implementation does not work and its not asked for question 2, so its left disabled. 
    temp = TRIAD(raw_accel_vector, raw_mag_vector, np.array([0, 0, 9.0665]), magnetic_north_normalized, returnRotMatrx=False)
    initial_offset = sm.UnitQuaternion()
    initial_offset.data[0] = qconj(temp.data[0])

    estimator.set_initial_conditions((raw_accel_vector, raw_mag_vector))
    #Set up 3D visualization
    xml_path = Path("resources") / "phone.xml"
    model = mj.MjModel.from_xml_path(str(xml_path))
    mujoco_model_data = mj.MjData(model)
    # To not repeat code, if for some reason you don't want to watch the awsome visualization and just want to see the boring plots, it will open the viewer and them immediately close it. 
    # I just don't know how else to do it without repeating the for loop outside the with statement. 
    with mujoco.viewer.launch_passive(model, mujoco_model_data) as viewer:
        if not do_3D_vis:
            viewer.close()
        # Compute initial orientation from TRIAD or whatever
        viewer.cam.distance *= 300.0

        joint_name = "free_joint"
        joint_id = model.joint(joint_name).id
        qpos_addr = model.jnt_qposadr[joint_id]

        v_a_hat_joint_name = "v_a_hat_joint"
        v_a_hat_joint_id = model.joint(v_a_hat_joint_name).id
        qpos_addr_v_a_hat = model.jnt_qposadr[v_a_hat_joint_id]

        v_m_hat_joint_name = "v_m_hat_joint"
        v_m_hat_joint_id = model.joint(v_m_hat_joint_name).id
        qpos_addr_v_m_hat = model.jnt_qposadr[v_m_hat_joint_id]


        # Optinally show or hide things. 
        if not show_extra_vectors:
            model.geom_rgba[model.geom(name="v_m_hat").id][3] = 0.0
            model.geom_rgba[model.geom(name="v_a_hat").id][3] = 0.0

        if not show_spatial_coords:
            model.geom_rgba[model.geom(name="x_axis").id][3] = 0.0
            model.geom_rgba[model.geom(name="y_axis").id][3] = 0.0
            model.geom_rgba[model.geom(name="z_axis").id][3] = 0.0

        if not show_body_coords:
            model.geom_rgba[model.geom(name="body_x_axis").id][3] = 0.0
            model.geom_rgba[model.geom(name="body_y_axis").id][3] = 0.0
            model.geom_rgba[model.geom(name="body_z_axis").id][3] = 0.0
            


        start_time = time.time()
        time_simulated = 0.0
        # Process the sensor data and update the Mahony filter
        for index, row in csv_data.iterrows():
            # Extract measurements from csv data. 
            curr_time = row['t']
            raw_mag_vector = np.array([row['mx'], row['my'], row['mz']])
            raw_gyro_vector = np.array([row['gyrox'], row['gyroy'], row['gyroz']])
            raw_accel_vector = np.array([row['ax'], row['ay'], row['az']])

            # Perform the calculations. 
            current_orientation_quat = estimator.time_step(raw_mag_vector, raw_gyro_vector, raw_accel_vector)
            # Save the results. 
            times.append(curr_time)
            rotation_angles.append(tr2angvec(current_orientation_quat.R)[0])
            euler_angles = SO3.UnitQuaternion(current_orientation_quat).rpy(unit='deg')
            roll.append(euler_angles[0])
            pitch.append(euler_angles[1])
            yaw.append(euler_angles[2])


            if estimator.get_estimated_error is not None :
                error_estimates.append(estimator.get_estimated_error)
            if estimator.get_bias is not None:
                bias_estimates.append(np.linalg.norm(estimator.get_bias))



            if do_3D_vis:
                # Update the model with the new orientation
                #phone mesh
                time_simulated += time_step
                mujoco_model_data.qpos[qpos_addr:qpos_addr+3] = [0,0,0]
                print()
                mujoco_model_data.qpos[qpos_addr+3:qpos_addr+7] = (initial_offset * current_orientation_quat).data[0]


                # TODO more fun visualization, it seems like v_hat_m is not pointing in the right direction. 
                # estimate vectors
                # mujoco_model_data.qpos[qpos_addr_v_a_hat:qpos_addr_v_a_hat+4] = get_quat_from_vec(mahony_filter.v_hat_a)
                mujoco_model_data.qpos[qpos_addr_v_m_hat:qpos_addr_v_m_hat+4] = get_quat_from_vec(estimator.get_v_hat_m, negate_z=False)
                mujoco_model_data.qpos[qpos_addr_v_a_hat:qpos_addr_v_a_hat+4] = get_quat_from_vec(estimator.get_v_hat_a, negate_z=True)
                mujoco.mj_forward(model, mujoco_model_data)# This is called pre sleep so we use part of our time step to update the viewer, but this wont be been unil viewer.synyc() is called.
                
                if not viewer.is_running():
                    print("Viewer is closed. Exiting...")
                    break
                    
                # Calculate the time to sleep
                elasped_time = time.time() - start_time
                sleep_time = time_simulated - elasped_time# if this is negative it means that the calculations are taking longer than the time step they are simulating so the simulation will be delayed. 
                if sleep_time > 0:
                    time.sleep(sleep_time)# Sleep enough such that the real time elapsed matches the simlated time elapsed. 
                else:
                    print(f"Warning, simulation is delayed: {-sleep_time * 1000:.2f} ms")

                viewer.sync()
    return times, rotation_angles, bias_estimates, error_estimates, roll, pitch, yaw