import argparse
from itertools import count

import gym
import scipy.optimize
from torch.autograd import Variable
import torch

from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
#from trpo import trpo_step
from utils import trpo_functions
from arguments import arg_parse_TRPO

import trpo
import numpy as np

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

###########################################################
""" check this argument, gives error without this/ gives wrong value"""
torch.set_default_tensor_type('torch.DoubleTensor')
###########################################################

# torch stuff
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def main():
    args = arg_parse_TRPO()
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    running_state = ZFilter((num_inputs,), clip=5)
    running_reward = ZFilter((1,), demean=False, clip=10)

    policy_net = Policy(num_inputs, num_actions)
    value_net = Value(num_inputs)
    train(args, env, policy_net, value_net, running_state)



def select_action(state, policy_net):
    state = torch.tensor(state).unsqueeze(0).to(device)
    #state = torch.from_numpy(state).unsqueeze(0).to(device)
    action_mean, _, action_std = policy_net(torch.autograd.Variable(state))
    action = torch.normal(action_mean, action_std)
    return action




def train(args, env, policy_net, value_net, running_state):
    for i_episode in count(1):
        memory = Memory()

        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        while num_steps < args.batch_size:
            state = env.reset()
            state = running_state(state)

            reward_sum = 0
            for t in range(10000): # Don't infinite loop while learning
                action = select_action(state, policy_net)
                action = action.data[0].numpy()
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward

                next_state = running_state(next_state)

                mask = 1
                if done:
                    mask = 0

                memory.push(state, np.array([action]), mask, next_state, reward)

                if args.render:
                    env.render()
                if done:
                    break

                state = next_state
            num_steps += (t-1)
            num_episodes += 1
            reward_batch += reward_sum

        reward_batch /= num_episodes
        batch = memory.sample()

        #########################
        # TRPO update parameters
        #########################
        update_trpo = trpo.update_params(batch, value_net, policy_net, args, trpo_functions)
        update_trpo.execute()

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum, reward_batch))

if __name__ == '__main__':
    main()
