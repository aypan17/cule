import numpy as np 
import torch

def proxy_reward(rew, ram, cached_ram, diver_bonus=0, o2_pen=0, bullet_pen=0, space_reward=False):
    """ Calculates proxy reward from cached ram and ram, both of size (ales, 128) with torch.uint8. 
        "seaquest": dict(enemy_obstacle_x=range(30, 34),
                     player_x=70,
                     player_y=97,
                     diver_or_enemy_missile_x=range(71, 75),
                     player_direction=86,
                     player_missile_direction=87,
                     oxygen_meter_value=102,
                     player_missile_x=103,
                     score=[57, 58],
                     num_lives=59,
                     divers_collected_count=62) 
    """

    # Score bonus
    # reward = 1000 * ((ram[:,57] // 16) - (cached_ram[:,57] // 16)) + 100 * ((ram[:,57] % 16) - (cached_ram[:,57] % 16)) + \
    #         10 * ((ram[:,58] // 16) - (cached_ram[:,58] // 16)) + ((ram[:,58] % 16) - (cached_ram[:,58] % 16))

    # if any(reward < 0):
    #     print(ram[:,57:59])
    #     print(cached_ram[:,57:59])
    #     assert False
    
    # Space reward
    # Seaquest: xmax = 134, ymax = 108
    if space_reward:
        rew *= (ram[:,97] < 54).to(dtype=torch.float32)

    
    #reward = reward.to(dtype=torch.float32)
    ram = ram.to(dtype=torch.float32)
    cached_ram = ram.to(dtype=torch.float32)

    # Diver bonus
    # ReLU for faster zeroing of negative values
    reward = diver_bonus * torch.nn.functional.relu(ram[:,62] - cached_ram[:,62], inplace=True)

    # O2 penalty
    reward -= o2_pen * (ram[:,102] < 16).to(dtype=torch.float32)   

    # Bullet penalty
    reward -= bullet_pen * ((cached_ram[:,103] == 0) & (ram[:,103] != 0)).to(dtype=torch.float32)

    return reward + rew