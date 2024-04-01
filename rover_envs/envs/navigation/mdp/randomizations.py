
from typing import TYPE_CHECKING  # noqa: F401

import torch
from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.envs import BaseEnv
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.envs import RLTaskEnv
from ..utils.terrains.terrain_importer import ExomyTerrainImporter


def reset_root_state_rover(env: BaseEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, z_offset: float = 0.5):
    """
    Generate random root states for the rovers, based on terrain_based_spawn_locations.
    """
    # Get the rover asset
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get the terrain and sample new spawn locations
    terrain: ExomyTerrainImporter = env.scene.terrain
    spawn_locations = terrain.get_spawn_locations()
    
    try:
        spawn_index = torch.randperm(spawn_locations.size(0), device=env.device)[:len(env_ids)]
    except:
        spawn_index = torch.randperm(spawn_locations.size(0), device=env.device)[:env.num_envs]

    spawn_locations = spawn_locations[spawn_index]

    # Add a small z offset to the spawn locations to avoid spawning the rover inside the terrain.
    positions = spawn_locations
    positions[:, 2] += z_offset

    #print(f"Env_ids: {env_ids}")
    #print(f"env.num_envs: {env.num_envs}")

    # Random angle
    try:
        angle = torch.rand(len(env_ids), device=env.device) * 2 * torch.pi
        quat = torch.zeros(len(env_ids), 4, device=env.device)
        quat[:, 0] = torch.cos(angle / 2)
        quat[:, 3] = torch.sin(angle / 2)
        orientations = quat
    except TypeError:
        print("env_ids is None. Cannot proceed with calculations.")
        #orientations = torch.zeros_like(positions)  # or any other appropriate handling
        orientations = torch.zeros(env.num_envs, 4, device=env.device)

    # Update the environment origins, so that the terrain targets are sampled around the new origin.
    #env.scene.terrain.env_origins[env_ids] = positions
    
    # Set the root state     
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
