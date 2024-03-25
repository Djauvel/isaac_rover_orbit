import os

from omni.isaac.orbit import sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg
from rover_envs.envs.navigation.utils.terrains.terrain_importer import ExomyTerrainImporter
from omni.isaac.orbit.utils.configclass import configclass

@configclass
class ExoMyTerrainSceneCfg(InteractiveSceneCfg):
    # Ground Terrain
    terrain = TerrainImporterCfg(
        class_type=ExomyTerrainImporter,
        prim_path="/World/terrain",
        terrain_type="usd",
        collision_group=-1,
        usd_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "environment_usd/ExoMyTerrain_Large.usd",
            
        ),
    )