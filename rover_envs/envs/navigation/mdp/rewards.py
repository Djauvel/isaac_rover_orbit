from __future__ import annotations

from typing import TYPE_CHECKING

import torch
# Importing necessary modules from the omni.isaac.orbit package
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def distance_to_target_reward(env: RLTaskEnv, command_name: str) -> torch.Tensor:
    """
    Calculate and return the distance to the target.

    This function computes the Euclidean distance between the rover and the target.
    It then calculates a reward based on this distance, which is inversely proportional
    to the squared distance. The reward is also normalized by the maximum episode length.
    """

    # Accessing the target's position through the command manage,
    # we get the target position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Calculating the distance and the reward
    distance = torch.norm(target_position, p=2, dim=-1)

    # Return the reward, normalized by the maximum episode length
    #print(f"NORMALIZED REWARD DISTTOTARGET: {(1.0 / (1.0 + (0.11 * distance * distance))) / env.max_episode_length}")
    #return (1.0 / (1.0 + (0.11 * distance * distance)))/ env.max_episode_length
    return torch.exp(-distance*2)


def reached_target(env: RLTaskEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the rover is within a certain threshold distance from the target.
    If the target is reached, a scaled reward is returned based on the remaining time steps.
    """

    # Accessing the target's position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)
    time_steps_to_goal = env.max_episode_length - env.episode_length_buf
    reward_scale = time_steps_to_goal / env.max_episode_length

    #print(f"NORMALIZED REACHED TARGET REWARD: {1.0 * reward_scale}, CONDITION {distance < threshold}")
    # Return the reward, scaled depending on the remaining time steps
    return torch.where(distance < threshold, 1.0 * reward_scale, 0)
    #Fixed Reward
    #return torch.where(distance < threshold, 1.0, 0)


def oscillation_penalty(env: RLTaskEnv) -> torch.Tensor:
    """
    Calculate the oscillation penalty.

    This function penalizes the rover for oscillatory movements by comparing the difference
    in consecutive actions. If the difference exceeds a threshold, a squared penalty is applied.
    """
    # Accessing the rover's actions
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    # Calculating differences between consecutive actions
    linear_diff = action[:, 1] - prev_action[:, 1]
    angular_diff = action[:, 0] - prev_action[:, 0]

    # TODO combine these 5 lines into two lines
    angular_penalty = torch.where(angular_diff*3 > 0.05, torch.square(angular_diff*3), 0.0)
    linear_penalty = torch.where(linear_diff*3 > 0.05, torch.square(linear_diff*3), 0.0)

    angular_penalty = torch.pow(angular_penalty, 2)
    linear_penalty = torch.pow(linear_penalty, 2)
    #print(f"OSCILLATION PENALTY: {(angular_penalty + linear_penalty) / env.max_episode_length}")
    return (angular_penalty + linear_penalty) / env.max_episode_length


def angle_to_target_reward(env: RLTaskEnv, command_name: str) -> torch.Tensor:
    """
    Calculate the penalty for the angle between the rover and the target.

    This function computes the angle between the rover's heading direction and the direction
    towards the target. A penalty is applied if this angle exceeds a certain threshold.
    """

    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]

    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])

    #print(f"ANGLE TO TARGET PENALTY: {angle}")
    # Return the absolute value of the angle, normalized by the maximum episode length.
    #return torch.where(torch.abs(angle) > 2.0, torch.abs(angle), 0.0)
    #print(f"Bool ANGLE: {(angle < -1) & (angle > -2)}")
    #print(f"ANGLE TO TARGET REWARD:{1.0 / env.max_episode_length}, CONDITION {(angle < -1.2) & (angle > -1.8)}")
    #return torch.where((angle < -1.2) & (angle > -1.8),1.0 ,0.0)
    return torch.where(((angle < -1.4955) & (angle > -1.6445)),1.0,0.0)

def angle_to_target_penalty(env: RLTaskEnv, command_name: str) -> torch.Tensor:
    """
    Calculate the penalty for the angle between the rover and the target.

    This function computes the angle between the rover's heading direction and the direction
    towards the target. A penalty is applied if this angle exceeds a certain threshold.
    """

    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]

    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])

    # Return the absolute value of the angle, normalized by the maximum episode length.
    #print(f"ANGLE TO TARGET PENALTY: {angle}, CONDITION {((angle < -1.14) & (angle > -2))}\n")
    # 24 degrees to each side
    return torch.where(((angle > -1.3955) | (angle < -1.7445)),1.0,0.0)

def heading_soft_contraint_rev(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate a penalty for driving backwards.

    This function applies a penalty when the rover's action indicates reverse movement.
    The penalty is normalized by the maximum episode length.
    """
    #print(f"HEADING SOFT CONSTRAINT: {1.0/ env.max_episode_length}, CONDITION: {env.action_manager.action[:, 0] < 0.0}")
    return torch.where(env.action_manager.action[:, 0] < 0.0, 1.0, 0.0)

def collision_penalty(env: RLTaskEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    Calculate a penalty for collisions detected by the sensor.

    This function checks for forces registered by the rover's contact sensor.
    If the total force exceeds a certain threshold, it indicates a collision,
    and a penalty is applied.
    """
    # Accessing the contact sensor and its data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    force_matrix = contact_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)
    # Calculating the force and applying a penalty if collision forces are detected
    normalized_forces = torch.norm(force_matrix, dim=1)
    forces_active = torch.sum(normalized_forces, dim=-1) > 1
    return torch.where(forces_active, 1.0, 0.0)

def far_from_target_reward(env: RLTaskEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Gives a penalty if the rover is too far from the target.
    """

    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    distance = torch.norm(target_position, p=2, dim=-1)

    #print(f"FAR FROM TARGET PENALTY: {1.0/ env.max_episode_length}, CONDITION {distance > threshold}")

    return torch.where(distance > threshold, 1.0, 0.0)

# ERC specific rewards and penalties 
def danger_landmark_penalty(env: RLTaskEnv, threshold: float, marker_position) -> torch.Tensor:
    """
    Gives a penalty if the rover comes too close to a danger landmark
    """
    
    rover_asset: RigidObject = env.scene["robot"]
    rover_position = rover_asset.data.root_state_w[:, :3]

    marker_position_tensor = torch.tensor(marker_position, dtype=torch.float32, device=rover_position.device)

    # Calculate distance
    distance: torch.Tensor = torch.norm(marker_position_tensor - rover_position, p=2, dim=-1)
    #print(f"Distance to Danger: {distance}")
    return torch.where(distance < threshold, 1.0 / env.max_episode_length, 0.0)

def time_penalty(env: RLTaskEnv, time_penalty : float) -> torch.Tensor:
    """Reward function for the reinforcement learning task"""
    # Apply time penalty
    return torch.tensor(time_penalty, dtype=torch.float32)

def falling_penalty(env: RLTaskEnv, low_bound : float, high_bound : float) -> torch.Tensor:
    """
    Checks whether the rover has fallen off the environment based on the orientation of the rover.
    """
    rover_asset: RigidObject = env.scene["robot"]
    roll, pitch, _ = euler_xyz_from_quat(rover_asset.data.root_state_w[:, 3:7])

    return torch.where(((low_bound < pitch) & (pitch < high_bound)) | ((low_bound < roll) & (roll < high_bound)), 1.0/ env.max_episode_length, 0.0)