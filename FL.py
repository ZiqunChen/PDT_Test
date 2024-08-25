import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import torch
from dataset import Dataset
from net import Transformer
import common_args
import argparse
import copy
from FL_train import local_train
from utils import (
    build_bandit_data_filename,
    build_bandit_model_filename,
    worker_init_fn,
    FedAvg
)

import numpy as np
import random





if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)

    parser.add_argument('--seed', type=int, default=0)

    args = vars(parser.parse_args())
    # args = vars(parser.parse_args(args=[]))
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    num_epochs = args['num_epochs']
    seed = args['seed']
    lin_d = args['lin_d']
    
    tmp_seed = seed
    
    if seed == -1:
        tmp_seed = 0

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    dataset_config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
    }    
    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
    }    

    
    state_dim = 1
    dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
    path_train = build_bandit_data_filename(
    env, n_envs, dataset_config, mode=0)
    model_config.update({'var': var, 'cov': cov})
    
    
    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': True,
    }
    global_model = Transformer(config).to(device)
    
    params = {
        'batch_size': 64,
        'shuffle': True,
    }
    
    
    train_dataset = Dataset(path_train, config)


    dataloader = torch.utils.data.DataLoader(train_dataset, **params)
   
    num_rounds = 1000
    num_clients = 5
    
    
    for round in range(num_rounds):
        # Distribute global model to clients
        client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
        
        # Local training on each client
        local_model_states = [local_train(client_model, dataloader) for client_model in client_models]
        
        # Aggregate updates and update global model
        global_model = FedAvg(global_model, local_model_states)
    
    filename = build_bandit_model_filename(env, model_config)
    torch.save(global_model.state_dict(), f'FL_local_models/{filename}.pt')
