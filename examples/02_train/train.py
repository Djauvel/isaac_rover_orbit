import argparse
import math
import os
import random
from datetime import datetime
import wandb

import gymnasium as gym
from omni.isaac.orbit.app import AppLauncher
import wandb.wandb_agent

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ExomySandbox-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--agent", type=str, default="PPO", help="Name of the agent.")
#------------ ERC -----------------
parser.add_argument("--lr", type=float, default=0.0, help="Learning rate")
parser.add_argument("--MLP_layers", type=list, default=[], help="Multilayer perceptron layers")
parser.add_argument("--hl", type=int, default=0, help="Horizon Length / Rollouts")
parser.add_argument("--kl", type=float, default=0.0, help="KL divergence threshold")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"

app_launcher = AppLauncher(launcher_args=args_cli, experience=app_experience)
simulation_app = app_launcher.app

from omni.isaac.orbit.envs import RLTaskEnv  # noqa: E402
from omni.isaac.orbit.utils.dict import print_dict  # noqa: E402
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml  # noqa: E402


def log_setup(experiment_cfg, env_cfg, agent):
    """
    Setup the logging for the experiment.

    Note:
        Copied from the ORBIT framework.
    """
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs
    log_dir = datetime.now().strftime("%b%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir = f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'

    log_dir += f"_{agent}"

    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir

    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)
    return log_dir


def video_record(env: RLTaskEnv, log_dir: str, video: bool, video_length: int, video_interval: int) -> RLTaskEnv:
    """
    Function to check and setup video recording.

    Note:
        Copied from the ORBIT framework.

    Args:
        env (RLTaskEnv): The environment.
        log_dir (str): The log directory.
        video (bool): Whether or not to record videos.
        video_length (int): The length of the video (in steps).
        video_interval (int): The interval between video recordings (in steps).

    Returns:
        RLTaskEnv: The environment.
    """

    if video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % video_interval == 0,
            "video_length": video_length,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        return gym.wrappers.RecordVideo(env, **video_kwargs)

    return env


from omni.isaac.orbit_tasks.utils import parse_env_cfg  # noqa: E402
from skrl.utils import set_seed  # noqa: E402

import rover_envs.envs.navigation.robots.aau_rover  # noqa: E402, F401
# Import agents
from rover_envs.learning.train import get_agent  # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402
from rover_envs.utils.skrl_utils import SkrlSequentialLogTrainer  # noqa: E402
from rover_envs.utils.skrl_utils import SkrlVecEnvWrapper  # noqa: E402
from collections import OrderedDict
from torch.nn import ModuleList

#Make experiment_cfg global

sweep_configuration = {
            'method': 'bayes',
            'name': 'sweep',
            'metric': {'goal': 'maximize', 'name': 'Reward / Total reward (mean)'},
            'parameters': 
            {
                'lr': {'max': 1.e-2, 'min': 1.e-5},
                'kl': {'max':0.024, 'min':0.002},
                'hl': {'max':128,'min':32}, #aka rollouts
                'MLP_layers' : {"values":[[256,128,64],
                                        [512,256,128],
                                        [512,512,256,128,64],
                                        [1024,512,512,256,128,64]]}
            }
        }

def train():
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    wandb.init(
    project='isaac_rover_orbit_ERC',  # Specify your project name
    entity='g666',    # Specify your username or team name
    name=f'ERC_{time_str}',  # Specify a name for your experiment
    notes='ERC_TEST',  # Add notes or description
    tags=['reinforcement-learning', 'agent-training'],  # Add relevant tags
    sync_tensorboard=True,
    )

    args_cli_seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

    experiment_cfg = parse_skrl_cfg(args_cli.task + f"_{args_cli.agent}")

    experiment_cfg['agent']['learning_rate'] = args_cli.lr
    experiment_cfg['agent']['kl_threshold'] = args_cli.kl
    experiment_cfg['agent']['rollouts'] = args_cli.hl
    mlp_layers = args_cli.MLP_layers

    # Remove characters that are not digits, commas, or spaces
    cleaned_layers = [x for x in mlp_layers if x.isdigit() or x == ',' or x == ' ']

    # Join digits to form integers, ignoring commas
    formatted_arg = [int(''.join(group)) for group in ''.join(cleaned_layers).split(',') if group.strip().isdigit()]

    experiment_cfg['agent']['mlp'] = formatted_arg
    
    # Manual override for mlp layers
    #experiment_cfg['agent']['mlp'] = [1024,512,512,256,128,64]

    print(f"Initializing training with Hyperparameters:") 
    print(f"kl-threshold: {experiment_cfg['agent']['kl_threshold']}")
    print(f"learning-rate: {experiment_cfg['agent']['learning_rate']}")
    print(f"horizon-length: {experiment_cfg['agent']['rollouts']}")
    print(f"MLP layers: {experiment_cfg['agent']['mlp']}")
    
    log_dir = log_setup(experiment_cfg, env_cfg, args_cli.agent)

    # Create the environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless, viewport=args_cli.video)
    # Check if video recording is enabled
    env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
    # Wrap the environment
    env: RLTaskEnv = SkrlVecEnvWrapper(env)
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # Get the observation and action spaces
    num_obs = env.observation_manager.group_obs_dim["policy"][0]
    num_actions = env.action_manager.action_term_dim[0]
    
    observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))

    # Set a custom number of timesteps
    trainer_cfg = experiment_cfg["trainer"]
    #trainer_cfg["timesteps"] = 1000000

    #Placeholder value
    mlp_layers = experiment_cfg["agent"]["mlp"]

    agent = get_agent(args_cli.agent, env, observation_space, action_space, experiment_cfg, mlp_layers)

    trainer = SkrlSequentialLogTrainer(cfg=trainer_cfg, agents=agent, env=env)
    trainer.train()

    wandb.teardown()
    env.close()
    simulation_app.close()

def train_sweeps():
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    wandb.init(
    project='isaac_rover_orbit_ERC',  # Specify your project name
    entity='g666',    # Specify your username or team name
    name=f'ERC_{time_str}',  # Specify a name for your experiment
    notes='ERC_TEST',  # Add notes or description
    tags=['reinforcement-learning', 'agent-training'],  # Add relevant tags
    sync_tensorboard=True,
    )

    train()

    

#def sweep():


    #api = wandb.Api()
    #sweep = api.sweep(path=f'g666/isaac_rover_orbit_ERC/{sweep_id}')
    
    #print(f"WANDB CONFIG PARAMETERS: {sweep}")
    #print(f"WANDB CONFIG PARAMETERS: {wandb.config}")
    #print(f"WANDB CONFIG PARAMETERS: {wandb.Api.sweep}")
    #m = wandb.config.parameters["lr"]
    #print(f"lr : {m}")
    
    #while wandb.config:
    #    try:
    #        experiment_cfg["agent"]["learning_rate"] = wandb.config.lr
    #        experiment_cfg["agent"]["kl_threshold"] = wandb.config.kl
    #        experiment_cfg["agent"]["rollouts"] = wandb.config.hl
    #        experiment_cfg["agent"]["mlp"] = wandb.config.MLP_layers
    #    except:
    #        print("Not ready yet")

    #train()



if __name__ == "__main__": 
    # Run the sweep
    #train_sweeps()

    # normal training
    train()