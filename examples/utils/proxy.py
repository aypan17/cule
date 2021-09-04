import numpy as np 
import torch

def proxy_reward(rew, cached_rew, ram, cached_ram, diver_bonus=0, prox_bonus=0, o2_pen=0, lives_pen=0, bullet_pen=0, space_reward=False):
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

        "riverraid": dict(player_x=51,
                      missile_x=117,
                      missile_y=50,
                      fuel_meter_high=55,  # high value displayed
                      fuel_meter_low=56  # low value
                      ),

        # The player scores points for shooting enemy tankers (30 points), helicopters (60 points), fuel depots (80 points), jets (100 points), and bridges (500 points). The jet refuels when it flies over a fuel depot.
    """

    # Score bonus
    # reward = 1000 * ((ram[:,57] // 16) - (cached_ram[:,57] // 16)) + 100 * ((ram[:,57] % 16) - (cached_ram[:,57] % 16)) + \
    #         10 * ((ram[:,58] // 16) - (cached_ram[:,58] // 16)) + ((ram[:,58] % 16) - (cached_ram[:,58] % 16))

    # if any(reward < 0):
    #     print(ram[:,57:59])
    #     print(cached_ram[:,57:59])
    #     assert False
    
    # Space reward
    ram = ram.to(dtype=torch.float32)
    cached_ram = ram.to(dtype=torch.float32)

    # Space reward
    # Seaquest: xmax = 134, ymax = 108
    # River raid: xmax = 110, xmin = 41
    reward = -1 * space_reward * ((ram[:,51] < 50) | (ram[:,51] > 100)).to(dtype=torch.float32)

    # Diver bonus
    # ReLU for faster zeroing of negative values
    #reward += diver_bonus * torch.nn.functional.relu(ram[:,62] - cached_ram[:,62])

    # Proximity bonus
    reward += prox_bonus * (5 * (ram[:,70] - cached_ram[:,70]) + torch.sum(cached_ram[:,71:76] - ram[:,71:76], 1))

    # Lives penalty
    reward += lives_pen * (ram[:,59] - cached_ram[:,59])

    # O2 penalty
    reward -= o2_pen * (ram[:,102] < 16).to(dtype=torch.float32) * (ram[:,102] - cached_ram[:, 62])

    # Bullet penalty
    reward -= bullet_pen * ((cached_ram[:,103] == 0) & (ram[:,103] != 0)).to(dtype=torch.float32)

    return reward.to(rew.device) + rew