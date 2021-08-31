import argparse
import os
import gym 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

import torch
import torch.nn.functional as F

from utils.openai.envs import create_vectorize_atari_env

_path = os.path.abspath(os.path.pardir)
sys.path = [os.path.join(_path, 'a2c')] + sys.path
from model import ActorCritic

def test(args, model, env):

    width, height = 84, 84
    num_ales = 1

    observation = torch.from_numpy(env.reset()).squeeze(1)

    lengths = torch.zeros(num_ales, dtype=torch.int32)
    rewards = torch.zeros(num_ales, dtype=torch.float32)
    all_done = torch.zeros(num_ales, dtype=torch.bool)
    not_done = torch.ones(num_ales, dtype=torch.bool)

    fire_reset = torch.zeros(num_ales, dtype=torch.bool)
    actions = torch.ones(num_ales, dtype=torch.uint8)

    maybe_npy = lambda a: a.numpy() 

    info = env.step(maybe_npy(actions))[-1]
    lives = torch.IntTensor([d['ale.lives'] for d in info])

    states = torch.zeros((num_ales, args.num_stack, width, height), dtype=torch.float32)
    states[:, -1] = observation.to(dtype=torch.float32)

    while not all_done.all():
        logit = model(states)[1]

        actions = F.softmax(logit, dim=1).multinomial(1).cpu()
        actions[fire_reset] = 1

        env.render()
        observation, reward, done, info = env.step(maybe_npy(actions))

        # convert back to pytorch tensors
        observation = torch.from_numpy(observation)
        reward = torch.from_numpy(reward.astype(np.float32))
        done = torch.from_numpy(done.astype(np.bool))
        new_lives = torch.IntTensor([d['ale.lives'] for d in info])

        fire_reset = new_lives < lives
        lives.copy_(new_lives)

        observation = observation.to(dtype=torch.float32)

        states[:, :-1].copy_(states[:, 1:].clone())
        states *= (1.0 - done).view(-1, *[1] * (observation.dim() - 1))
        states[:, -1].copy_(observation.view(-1, *states.size()[-2:]))

        # update episodic reward counters
        lengths += not_done.int()
        rewards += reward.cpu() * not_done.float().cpu()

        all_done |= done.cpu()
        all_done |= (lengths >= args.max_episode_length)
        not_done = (all_done == False).int()

    return lengths, rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CuLE')
    parser.add_argument('game', type=str, help='Atari ROM filename')
    parser.add_argument('--num_stack', type=int, default=4, help='number of images in a stack (default: 4)')
    parser.add_argument('--max_hostack', type=int, default=4, help='number of images in a stack (default: 4)')
    parser.add_argument('--num-stack', type=int, default=4, help='number of images in a stack (default: 4)')
    args = parser.parse_args()
    num_stack = args.num_stack

    env = create_vectorize_atari_env(args.game, 0, 1, episode_life=False, clip_rewards=False)
    env.reset()
    env.eval()

    model = ActorCritic(num_stack, env.action_space)
    if args.model_path is not None:
        model.load(args.model_path)
    model.eval()

    test(args, model, env)
    env.close()


