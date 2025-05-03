import mujoco as mj
import numpy as np
import mujoco.viewer, time, os, itertools
from kinematics import DHKinematics
from kinematics import mini_bot_geometric_inverse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.spatial import KDTree
import pickle

# used angle_limits = [(-np.pi, np.pi), (-.610, 3.575), (-.942, np.pi), (-np.pi, np.pi), (-2.58, 2.58), (-np.pi, np.pi)]
# to create the 2.4M point cloud. 
def animate_range_of_motion(angle_limits, model, mujoco_model_data):
    limits = np.array(angle_limits)
    mins, maxs = limits[:,0], limits[:,1]
    mids       = (mins + maxs) / 2.0
    amps       = (maxs - mins) / 2.0

    # duration (s) for one joint to sweep from min→max→min
    period = 4.0  

    try:
        with mujoco.viewer.launch_passive(model, mujoco_model_data) as v:
            v.cam.distance *= 2.0

            # start all joints at their mid positions
            mujoco_model_data.qpos[:6] = mids
            mj.mj_forward(model, mujoco_model_data)
            v.sync()

            while v.is_running():
                # cycle through joints 0…5
                for joint_idx in range(6):
                    start = time.time()
                    while v.is_running() and time.time() - start < period:
                        t = time.time() - start
                        # sine from -1 to +1 over one period
                        theta = np.sin(2 * np.pi * t / period)

                        # reset all to mids, then override the active joint
                        angles = mids.copy()
                        angles[joint_idx] = mids[joint_idx] + amps[joint_idx] * theta

                        mujoco_model_data.qpos[:6] = angles
                        mj.mj_forward(model, mujoco_model_data)
                        v.sync()

                        time.sleep(0.002)

                    # after finishing this joint, snap it back to mid before next
                    mujoco_model_data.qpos[joint_idx] = mids[joint_idx]
                    mj.mj_forward(model, mujoco_model_data)
                    v.sync()

    except KeyboardInterrupt:
        print("Interrupted, closing viewer...")
        v.close()

def generate_joint_configs(angle_limits, num_steps):
    """
    Generate XYZ positions of the end-effector for all joint angle combinations (multithreaded).

    Parameters:
        angle_limits (list of tuple): List of (min, max) joint limits.
        num_steps (int or list of int): Number of steps per joint.
        kinematics: Must have a method `forward(joint_angles)` returning (x, y, z).
        max_workers (int): Number of threads to use.

    Returns:
        np.ndarray of shape (N, 3): End-effector positions.
    """
    if isinstance(num_steps, int):
        steps_per_joint = [num_steps] * len(angle_limits)
    else:
        if len(num_steps) != len(angle_limits):
            raise ValueError("num_steps must be a single int or a list with the same length as angle_limits.")
        steps_per_joint = num_steps

    # Discretize each joint range
    discretized_angles = [
        np.linspace(lo, hi, n) for (lo, hi), n in zip(angle_limits, steps_per_joint)
    ]

    # Generate all joint configurations (large list, but necessary for threading)
    joint_configs = list(itertools.product(*discretized_angles))



    return joint_configs

def compute_position_batch(joint_angles_batch, kinematics: DHKinematics):
    """Compute positions for a batch of joint angle configurations."""
    # Collect all results for the batch
    results = [kinematics.foreward(np.array(ja)).t for ja in joint_angles_batch]
    return results

def distance_favored_sampling(points, target_count=100_000, center=np.array([0, 0, 0]), power=5):
    """
    Takes in an existing point cloud and downsamples it according to an exponetial probability distribution. 
    a positive power value will retain samples from the edges of the sphere.  
    Retain more points far from the origin and fewer near the center.

    Args:
        points (np.ndarray): Nx3 array of point coordinates.
        target_count (int): Number of points to retain.
        center (np.ndarray): Reference point (e.g., robot base).
        power (float): Higher = sharper falloff toward the center.

    Returns:
        np.ndarray: Sampled point cloud.
    """
    distances = np.linalg.norm(points - center, axis=1)
    weights = distances**power
    probabilities = weights / weights.sum()
    indices = np.random.choice(len(points), size=target_count, replace=False, p=probabilities)
    return points[indices]

def generate_point_cloud(angle_limits, joint_steps, batch_size, kinematics, file=None)-> KDTree:
    joint_configs = generate_joint_configs(angle_limits, num_steps=joint_steps)
    print("Joint configs generated")
    max_workers = os.cpu_count()
    points = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(joint_configs), batch_size):
            batch = joint_configs[i:i + batch_size]  # Create a chunk of joint configurations
            futures.append(executor.submit(compute_position_batch, batch, kinematics))  # Submit the chunk
        print("Batches created, starting up...")        
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing positions"):
            batch_results = f.result()
            points.extend(batch_results)  # Flatten and add results to the final list 
    
    points = np.array(points)
    print("Computed", len(points), "end-effector positions.")
    print("Making tree")
    tree = KDTree(points)
    if file is not None:
        import pickle
        with open(file, 'wb') as f:
            pickle.dump(tree, f)
    return tree

with open("point_cloud_tree2.4M.pkl", 'rb') as f:
    tree = pickle.load(f)
def get_neighbor(point: np.ndarray):
    distance, index = tree.query(point)
    return distance, tree.data[index]