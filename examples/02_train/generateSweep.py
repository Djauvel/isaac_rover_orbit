#import wandb
#import wandb.wandb_agent
#import train
#import subprocess
#
#command = []
#
#sweep_configuration = {
#            'method': 'bayes',
#            'name': 'sweep',
#            'metric': {'goal': 'maximize', 'name': 'Reward / Total reward (mean)'},
#            'parameters': 
#            {
#                'lr': {'max': 1.e-2, 'min': 1.e-5},
#                'kl': {'max':0.024, 'min':0.002},
#                'hl': {'max':128,'min':32}, #aka rollouts
#                'MLP_layers' : {"values":[[256,128,64],
#                                        [512,256,128],
#                                        [512,512,256,128,64],
#                                        [1024,512,512,256,128,64]]}
#            }
#        }
#
#    
#sweep_id = wandb.sweep(sweep_configuration, project="isaac_rover_orbit_ERC")
#wandb.agent(sweep_id=sweep_id, project="isaac_rover_orbit_ERC",function=train.train())

mlp_layers = ['[', '5', '1', '2', ',', ' ', '2', '5', '6', ',', ' ', '1', '2', '8', ']']

# Remove characters that are not digits, commas, or spaces
cleaned_layers = [x for x in mlp_layers if x.isdigit() or x == ',' or x == ' ']

# Join digits to form integers, ignoring commas
formatted_arg = [int(''.join(group)) for group in ''.join(cleaned_layers).split(',') if group.strip().isdigit()]
print(formatted_arg)
