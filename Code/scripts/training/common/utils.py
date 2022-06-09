
import torch
from .logs_functions import logging


def epsilon_annealing(i_epsiode, max_episode, min_eps: float):
    # if i_epsiode --> max_episode, ret_eps --> min_eps
    # if i_epsiode --> 1, ret_eps --> 1
    slope = (min_eps - 1.0) / max_episode
    ret_eps = max(slope * i_epsiode + 1.0, min_eps)

    return ret_eps


def save_model(save_model,  env_name, training_method, agent, curiosity):

    if save_model == True:
        print("save_model: ", str(save_model))
        if training_method == "epsilon_greedy":
            if curiosity == 0:
                checkpoint_path_local, checkpoint_path_target = logging.create_checkpoint_path(
                    env_name, training_method)
            else:
                training_method_ = "ICM"
                checkpoint_path_local, checkpoint_path_target = logging.create_checkpoint_path(
                    env_name, training_method_)

        elif training_method == "greedy" or training_method == "MixIn" or training_method == "PBRS" or training_method == "DPBRS" or training_method == "GBRS":
            checkpoint_path_local, checkpoint_path_target = logging.create_checkpoint_path(
                env_name, training_method)

    torch.save(agent.q_local.state_dict(), checkpoint_path_local)
    torch.save(agent.q_target.state_dict(), checkpoint_path_target)


def create_log_files(training_method, env_name, deep_loging, curiosity, intrinsic_reward):

    log_f_deep = None
    log_int_re = None
    if training_method == "epsilon_greedy" or training_method == "greedy":
        if curiosity == 0:
            log_f_name, run_num = logging.create_log_files(
                env_name, training_method)
            log_f = open(log_f_name, "w+")
            log_f.write(
                'episode,timestep,steps_done,reward,avg_reward,eps\n')
        else:
            training_method_ = "ICM"
            log_f_name, run_num = logging.create_log_files(
                env_name, training_method_)
            log_f = open(log_f_name, "w+")
            log_f.write(
                'episode,timestep,steps_done,reward,avg_reward,eps\n')

    elif training_method == "PBRS" or training_method == "DPBRS" or training_method == "GBRS" or training_method == "MixIn":
        log_f_name, run_num = logging.create_log_files(
            env_name, training_method)
        log_f = open(log_f_name, "w+")
        if training_method == "MixIn":
            log_f.write(
                'episode,timestep,steps_done,reward,avg_reward,shaped_reward,shaped_avg_reward,eps,k\n')
        elif training_method == "PBRS" or training_method == "DPBRS":
            log_f.write(
                'episode,timestep,steps_done,reward,avg_reward,shaped_reward,shaped_avg_reward,eps,po_s\n')

        else:
            log_f.write(
                'episode,timestep,steps_done,reward,avg_reward,shaped_reward,shaped_avg_reward,eps\n')

    else:
        raise NotImplementedError(
            f'No training method named "{training_method}"')

    if deep_loging == True:
        if curiosity == 0:
            log_f_name_deep = logging.create_log_files_states_details(
                env_name, training_method)
            log_f_deep = open(log_f_name_deep, "w+")
            if training_method == "epsilon_greedy" or training_method == "greedy":
                log_f_deep.write(
                    "episode,timestep,reward,action,state,q_loss,icm_loss\n")
            else:
                log_f_deep.write(
                    "episode,timestep,reward,action,state,q_loss\n")
        else:
            training_method_ = "ICM"
            log_f_name_deep = logging.create_log_files_states_details(
                env_name, training_method_)
            log_f_deep = open(log_f_name_deep, "w+")
            log_f_deep.write(
                "episode,timestep,reward,action,state,q_loss,icm_loss\n")
    else:
        print("deep_logging: ", str(deep_loging))

    if intrinsic_reward == True:
        if curiosity == 0:
            log_f_name_int_re = logging.create_log_files_intrinsic_reward(
                env_name, training_method)
            log_int_re = open(log_f_name_int_re, "w+")
            if training_method == "epsilon_greedy" or training_method == "greedy":
                log_int_re.write(
                    "episode,timestep,done_steps,reward,int_reward\n")
            else:
                log_int_re.write(
                    "episode,timestep,done_steps,reward,int_reward,shaped_reward\n")

        else:
            training_method_ = "ICM"
            log_f_name_int_re = logging.create_log_files_intrinsic_reward(
                env_name, training_method_)
            log_int_re = open(log_f_name_deep, "w+")
            log_int_re.write(
                "episode,timestep,done_steps,reward,int_reward\n")
    else:
        print("calculate intrinsic reward ", str(intrinsic_reward))

    return run_num, log_f, log_f_deep, log_int_re
