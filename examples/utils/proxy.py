import numpy as np 

def proxy_reward(ram, cached_ram, diver_bonus=0, o2_pen=0, bullet_pen=0, space_reward=False):
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
    reward = 1000 * (ram[:57] >> 4 - cached_ram >> 4) + 100 * (ram[:57] % 16 - cached_ram % 16) + 10 * (ram[:58] >> 4 - cached_ram[:58] >> 4) + (ram[:58] % 16 - cached_ram[:58] % 16)

    # Space reward
    # Seaquest: xmax = 134, ymax = 108
    if space_reward:
    	if ram[97] < 54:
    		reward *= 0

    reward = reward.to(dtype=torch.float32)

    # Diver bonus
	reward += diver_bonus * np.min(ram[:62] - cached_ram[:62], 0)

    # O2 penalty
    reward -= o2_pen * (ram[:102] - cached_ram[:102])	

    # Bullet penalty
    reward -= bullet_pen * ((cached_ram[:103] == 0) & (ram[:103] != 0))

    return reward