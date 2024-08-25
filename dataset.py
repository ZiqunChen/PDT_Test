import pickle

import numpy as np
import torch

from utils import convert_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, path, config):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config

        # if path is not a list
        if not isinstance(path, list):
            path = [path]

        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)
            
        context_states = []
        context_actions = []
        context_next_states = []
        context_rewards = []
        query_states = []
        optimal_actions = []

        for traj in self.trajs:
            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            context_next_states.append(traj['context_next_states'])
            context_rewards.append(traj['context_rewards'])

            query_states.append(traj['query_state'])
            optimal_actions.append(traj['optimal_action'])

        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        context_next_states = np.array(context_next_states)
        context_rewards = np.array(context_rewards)
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        query_states = np.array(query_states)
        optimal_actions = np.array(optimal_actions)

        self.dataset = {
            'query_states': convert_to_tensor(query_states, store_gpu=self.store_gpu),
            'optimal_actions': convert_to_tensor(optimal_actions, store_gpu=self.store_gpu),
            'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
            'context_next_states': convert_to_tensor(context_next_states, store_gpu=self.store_gpu),
            'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
        }

        self.zeros = np.zeros(
            config['state_dim'] ** 2 + config['action_dim'] + 1
        )
        self.zeros = convert_to_tensor(self.zeros, store_gpu=self.store_gpu)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset['query_states'])

    def __getitem__(self, index):
        'Generates one sample of data'
        res = {
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }
        # 重新排列，用于增强模型的鲁棒性，特别是在训练时，它可以防止模型过拟合到固定的时间步序列。
        # 通过打乱时间步的顺序，模型在学习时必须依赖于各个时间步的具体特征，而不是某种固定的序列模式，从而提高模型的泛化能力。
        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]

        return res


