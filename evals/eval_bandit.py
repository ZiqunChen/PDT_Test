import matplotlib.pyplot as plt

import numpy as np
import scipy
import torch
from IPython import embed


from ctrls.ctrl_bandit import (
    BanditTransformerController,
    GreedyOptPolicy,
    EmpMeanPolicy,
    OptPolicy,
    PessMeanPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
)
from envs.bandit_env import BanditEnv, BanditEnvVec
from utils import convert_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deploy_online(env, controller, horizon):
    context_states = torch.zeros((1, horizon, env.dx)).float().to(device)
    context_actions = torch.zeros((1, horizon, env.du)).float().to(device)
    context_next_states = torch.zeros((1, horizon, env.dx)).float().to(device)
    context_rewards = torch.zeros((1, horizon, 1)).float().to(device)

    cum_means = []
    for h in range(horizon):
        batch = {
            'context_states': context_states[:, :h, :],
            'context_actions': context_actions[:, :h, :],
            'context_next_states': context_next_states[:, :h, :],
            'context_rewards': context_rewards[:, :h, :],
        }

        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = env.deploy(
            controller)

        context_states[0, h, :] = convert_to_tensor(states_lnr[0])
        context_actions[0, h, :] = convert_to_tensor(actions_lnr[0])
        context_next_states[0, h, :] = convert_to_tensor(next_states_lnr[0])
        context_rewards[0, h, :] = convert_to_tensor(rewards_lnr[0])

        actions = actions_lnr.flatten()
        mean = env.get_arm_value(actions)

        cum_means.append(mean)

    return np.array(cum_means)


def deploy_online_vec(vec_env, controller, horizon, include_meta=False):
    num_envs = vec_env.num_envs
    # context_states = torch.zeros((num_envs, horizon, vec_env.dx)).float().to(device)
    # context_actions = torch.zeros((num_envs, horizon, vec_env.du)).float().to(device)
    # context_next_states = torch.zeros((num_envs, horizon, vec_env.dx)).float().to(device)
    # context_rewards = torch.zeros((num_envs, horizon, 1)).float().to(device)

    context_states = np.zeros((num_envs, horizon, vec_env.dx))
    context_actions = np.zeros((num_envs, horizon, vec_env.du))
    context_next_states = np.zeros((num_envs, horizon, vec_env.dx))
    context_rewards = np.zeros((num_envs, horizon, 1))

    cum_means = []
    print("Deplying online vectorized...")
    for h in range(horizon):
        batch = {
            'context_states': context_states[:, :h, :],
            'context_actions': context_actions[:, :h, :],
            'context_next_states': context_next_states[:, :h, :],
            'context_rewards': context_rewards[:, :h, :],
        }

        controller.set_batch_numpy_vec(batch)

        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy(
            controller)

        context_states[:, h, :] = states_lnr
        context_actions[:, h, :] = actions_lnr
        context_next_states[:, h, :] = next_states_lnr
        context_rewards[:, h, :] = rewards_lnr[:,None]

        mean = vec_env.get_arm_value(actions_lnr)
        cum_means.append(mean)

    print("Deplyed online vectorized")
    
    cum_means = np.array(cum_means)
    if not include_meta:
        return cum_means
    else:
        meta = {
            'context_states': context_states,
            'context_actions': context_actions,
            'context_next_states': context_next_states,
            'context_rewards': context_rewards,
        }
        return cum_means, meta



def online(eval_trajs, model, n_eval, horizon, var, bandit_type):

    all_means = {}

    envs = []
    for i_eval in range(n_eval):
        print(f"Eval traj: {i_eval}")
        traj = eval_trajs[i_eval]
        means = traj['means']

        # TODO: Does bandit type need to be passed in?
        env = BanditEnv(means, horizon, var=var)
        envs.append(env)

    vec_env = BanditEnvVec(envs)
    
    controller = OptPolicy(
        envs,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T    
    assert cum_means.shape[0] == n_eval
    all_means['opt'] = cum_means


    controller = BanditTransformerController(
        model,
        sample=True,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['PDT'] = cum_means


    controller = EmpMeanPolicy(
        envs[0],
        online=True,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['Emp'] = cum_means

    controller = ThompsonSamplingPolicy(
        envs[0],
        std=var,
        sample=True,
        prior_mean=0.5,
        prior_var=1/12.0,
        warm_start=False,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['TS'] = cum_means
    
    controller = UCBPolicy(
        envs[0],
        const=1.0,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['UCB'] = cum_means

    all_means = {k: np.array(v) for k, v in all_means.items()}
    all_means_diff = {k: all_means['opt'] - v for k, v in all_means.items()}

    means = {k: np.mean(v, axis=0) for k, v in all_means_diff.items()}
    sems = {k: scipy.stats.sem(v, axis=0) for k, v in all_means_diff.items()}


    cumulative_regret = {k: np.cumsum(v, axis=1) for k, v in all_means_diff.items()}
    regret_means = {k: np.mean(v, axis=0) for k, v in cumulative_regret.items()}
    regret_sems = {k: scipy.stats.sem(v, axis=0) for k, v in cumulative_regret.items()}
    # np.save('regret_means_central_online.npy', regret_means)
    # np.save('regret_sems_central_online.npy', regret_sems)
    # fig, (ax2) = plt.subplots(1, 2, figsize=(8, 6))


    # for key in means.keys():
    #     if key == 'opt':
    #         ax1.plot(means[key], label=key, linestyle='--',
    #                 color='black', linewidth=2)
    #         ax1.fill_between(np.arange(horizon), means[key] - sems[key], means[key] + sems[key], alpha=0.2, color='black')
    #     else:
    #         ax1.plot(means[key], label=key)
    #         ax1.fill_between(np.arange(horizon), means[key] - sems[key], means[key] + sems[key], alpha=0.2)


    # ax1.set_yscale('log')
    # ax1.set_xlabel('Episodes')
    # ax1.set_ylabel('Suboptimality')
    # ax1.set_title('Online Evaluation')
    # ax1.legend()

    plt.figure(figsize=(8,6))
    for key in regret_means.keys():
        if key != 'opt':
            plt.plot(np.array(regret_means[key]), label=key)
            plt.fill_between(np.arange(horizon), np.array(regret_means[key] - regret_sems[key]), np.array(regret_means[key] + regret_sems[key]), alpha=0.2)
    # regret_means_central = np.load('regret_means_central_online.npy',allow_pickle=True).item()
    # regret_sems_central = np.load('regret_sems_central_online.npy',allow_pickle=True).item()
    # for key in regret_means_central.keys():
    #     if key != 'opt' and key == 'PDT':
    #         plt.plot(regret_means_central[key], label=key, color = 'orange')
    #         plt.fill_between(np.arange(horizon), regret_means_central[key] - regret_sems_central[key], regret_means_central[key] + regret_sems_central[key], alpha=0.2,color = 'orange')
    # ax2.set_yscale('log')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Regret')
    # plt.title('Regret Over Time')
    plt.legend()




def offline(eval_trajs, model, n_eval, horizon, var, bandit_type):
    all_rs_lnr = []
    all_rs_greedy = []
    all_rs_opt = []
    all_rs_emp = []
    all_rs_pess = []
    all_rs_thmp = []

    num_envs = len(eval_trajs)

    tmp_env = BanditEnv(eval_trajs[0]['means'], horizon, var=var)
    context_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_actions = np.zeros((num_envs, horizon, tmp_env.du))
    context_next_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_rewards = np.zeros((num_envs, horizon, 1))


    envs = []

    print(f"Evaling offline horizon: {horizon}")

    for i_eval in range(n_eval):
        # print(f"Eval traj: {i_eval}")
        traj = eval_trajs[i_eval]
        means = traj['means']

        # TODO: Does bandit type need to be passed in?
        env = BanditEnv(means, horizon, var=var)
        envs.append(env)

        context_states[i_eval, :, :] = traj['context_states'][:horizon]
        context_actions[i_eval, :, :] = traj['context_actions'][:horizon]
        context_next_states[i_eval, :, :] = traj['context_next_states'][:horizon]
        context_rewards[i_eval, :, :] = traj['context_rewards'][:horizon,None]


    vec_env = BanditEnvVec(envs)
    batch = {
        'context_states': context_states,
        'context_actions': context_actions,
        'context_next_states': context_next_states,
        'context_rewards': context_rewards,
    }

    opt_policy = OptPolicy(envs, batch_size=num_envs)
    emp_policy = EmpMeanPolicy(envs[0], online=False, batch_size=num_envs)
    lnr_policy = BanditTransformerController(model, sample=False, batch_size=num_envs)
    thomp_policy = ThompsonSamplingPolicy(
        envs[0],
        std=var,
        sample=False,
        prior_mean=0.5,
        prior_var=1/12.0,
        warm_start=False,
        batch_size=num_envs)
    lcb_policy = PessMeanPolicy(
        envs[0],
        const=.8,
        batch_size=len(envs))


    opt_policy.set_batch_numpy_vec(batch)
    emp_policy.set_batch_numpy_vec(batch)
    thomp_policy.set_batch_numpy_vec(batch)
    lcb_policy.set_batch_numpy_vec(batch)
    lnr_policy.set_batch_numpy_vec(batch)
    
    _, _, _, rs_opt = vec_env.deploy_eval(opt_policy)
    _, _, _, rs_emp = vec_env.deploy_eval(emp_policy)
    _, _, _, rs_lnr = vec_env.deploy_eval(lnr_policy)
    _, _, _, rs_lcb = vec_env.deploy_eval(lcb_policy)
    _, _, _, rs_thmp = vec_env.deploy_eval(thomp_policy)


    baselines = {
        'opt': np.array(rs_opt),
        'PDT': np.array(rs_lnr),
        'Emp': np.array(rs_emp),
        'TS': np.array(rs_thmp),
        'UCB': np.array(rs_lcb),
    }    
    baselines_means = {k: np.mean(v) for k, v in baselines.items()}
    colors = plt.cm.viridis(np.linspace(0, 1, len(baselines_means)))
    plt.bar(baselines_means.keys(), baselines_means.values(), color=colors)
    plt.title(f'Mean Reward on {n_eval} Trajectories')


    return baselines


def offline_graph(eval_trajs, model, n_eval, horizon, var, bandit_type):
    horizons = np.linspace(1, horizon, 50, dtype=int)

    all_means = []
    all_sems = []
    for h in horizons:
        config = {
            'horizon': h,
            'var': var,
            'n_eval': n_eval,
            'bandit_type': bandit_type,
        }
        config['horizon'] = h
        baselines = offline(eval_trajs, model, **config)
        plt.clf()

        means = {k: np.mean(v, axis=0) for k, v in baselines.items()}
        sems = {k: scipy.stats.sem(v, axis=0) for k, v in baselines.items()}
        all_means.append(means)

    plt.figure(figsize=(8,6))
    for key in means.keys():
        if not key == 'opt':
            regrets = [all_means[i]['opt'] - all_means[i][key] for i in range(len(horizons))]            
            plt.plot(horizons, np.array(regrets), label=key)
            plt.fill_between(horizons, np.array(regrets - sems[key]), np.array(regrets + sems[key]), alpha=0.2)

    # all_means_central = np.load('all_means_central_offline.npy',allow_pickle=True)
    # sems_central = np.load('sems_central_offline.npy',allow_pickle=True).item()
    # for key in means.keys():
    #     if not key == 'opt':
    #         regrets = [all_means_central[i]['opt'] - all_means_central[i][key] for i in range(len(horizons))]            
    #         plt.plot(horizons, regrets, label='PDT', color = 'orange')
    #         plt.fill_between(horizons, regrets - sems_central[key], regrets + sems_central[key], alpha=0.2, color = 'orange')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Dataset size')
    plt.ylabel('Suboptimality')
    config['horizon'] = horizon
