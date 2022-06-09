import time
from collections import deque
from dataclasses import dataclass


from typing import List


from logger_configs.loggers import result_logger, logger
import os
import numpy as np
import torch


from common.make_env import initialize_environment
from common.utils import epsilon_annealing, save_model, create_log_files
from common.statistics_plot import *
from common.training_methods import *
from common.arguments import train_args


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")


class Training:

    @dataclass
    class Hyperparameter:
        ENV: str
        BATCH_SIZE: int
        TAU: float
        GAMMA: float
        LEARNING_RATE: float
        TARGET_UPDATE: int
        buffer_size: int
        num_episodes: int
        print_every: int
        log_every: int
        hidden_dim: int
        min_eps: float
        max_eps_episode: int
        random_seed: float
        render: bool
        curiosity: int
        k: List[float]
        N: List[float]

    def __init__(self, hyperparams):

        self.env = None
        self.env_name = None
        self.agent = None
        self.thereshold = None
        self.policy = None
        self.training_method = None
        self.curiosity = None
        self.calculate_int_reward = None
        self.hyperparams = hyperparams
        self.steps_done = 0

######################################### Training Loop #########################################

    def train(self):

        scores_deque = deque(maxlen=100)
        scores_array = []
        avg_scores_array = []
        scores_deque_shaped = deque(maxlen=100)
        scores_array_shaped = []
        avg_scores_array_shaped = []

        max_episode_original_reward = None
        min_episode_original_reward = None
        max_100episodes_avg = None

        time_start = time.time()
        local_time = time.ctime(time_start)
        result_logger.debug("Initial Timestamp : {}".format(local_time))
        #training_method = args.training_method
        result_logger.debug(
            "training_Method : {}".format(self.training_method))

        if self.curiosity == 0:
            result_logger.debug("curiosity : no intrinsic curiosity")
        elif self.curiosity == 1:
            result_logger.debug(
                "curiosity : Intrinsic curiosity and extrinsic reward")
        elif self.curiosity == 2:
            result_logger.debug("curiosity : only Intrinsic curiosity")

        run_num, log_f, log_f_deep, log_int_re = create_log_files(
            self.training_method, self.env_name, args.deep_logging, self.curiosity, self.calculate_int_reward)

        logger.info("Starting model training for {} episodes.".format(
            self.hyperparams.num_episodes))

        for i_episode in range(self.hyperparams.num_episodes):

            #check training method to select between greedy or eps_greedy policy
            if self.training_method == "greedy" or self.training_method == "MixIn":
                eps = 0
            else:
                eps = epsilon_annealing(
                    i_episode, self.hyperparams.max_eps_episode, self.hyperparams.min_eps)

            #check training method and rund episode with/without RS
            if self.training_method == "epsilon_greedy" or self.training_method == "greedy":
                score, time_step, steps_done = Methods.run_episode(self.env_name,
                                                                   self.env, self.agent, self.steps_done,  eps, self.hyperparams.BATCH_SIZE, self.hyperparams.GAMMA, self.hyperparams.render, log_f_deep, i_episode, log_int_re)
            elif self.training_method == "MixIn":
                score, time_step, total_reward_shaped, steps_done = Methods.run_episode_with_MixInRandom_RS(self.env_name,
                                                                                                            self.env, self.agent, self.steps_done, eps, self.hyperparams.k[i_episode], self.hyperparams.BATCH_SIZE, self.hyperparams.GAMMA, self.hyperparams.render, log_f_deep, i_episode, log_int_re)
            elif self.training_method == "PBRS":
                score, time_step, total_reward_shaped, steps_done = Methods.run_episode_with_PBRS(self.env_name,
                                                                                                  self.env, self.agent, self.steps_done, eps, self.hyperparams.BATCH_SIZE, self.hyperparams.GAMMA, self.hyperparams.render, log_f_deep, i_episode, log_int_re)
            elif self.training_method == "DPBRS":
                score, time_step, total_reward_shaped, steps_done = Methods.run_episode_with_DynamicPBRS(self.env_name,
                                                                                                         self.env, self.agent, self.steps_done, eps, max_episode_original_reward, min_episode_original_reward, self.hyperparams.BATCH_SIZE, self.hyperparams.GAMMA, self.hyperparams.render, log_f_deep, i_episode, log_int_re)
            elif self.training_method == "GBRS":
                score, time_step, total_reward_shaped, steps_done = Methods.run_episode_with_GBRS_LL(self.env_name,
                                                                                                     self.env, self.agent, self.steps_done, eps, self.hyperparams.N[i_episode], self.hyperparams.BATCH_SIZE, self.hyperparams.GAMMA, self.hyperparams.render, log_f_deep, i_episode, log_int_re)
            else:
                raise NotImplementedError(
                    f'No training method named "{self.training_method}"')
            self.steps_done = steps_done
            scores_deque.append(score)
            scores_array.append(score)
            avg_score = np.mean(scores_deque)
            avg_scores_array.append(avg_score)
            dt = (int)(time.time() - time_start)

            if self.training_method == "PBRS" or self.training_method == "DPBRS" or self.training_method == "GBRS":
                scores_deque_shaped.append(total_reward_shaped)
                scores_array_shaped.append(total_reward_shaped)
                avg_score_shaped = np.mean(scores_deque_shaped)
                avg_scores_array_shaped.append(avg_score_shaped)
                if i_episode % self.hyperparams.print_every == 0 and i_episode > 0:
                    logger.debug('Episode: {:5} Timesteps: {} True_Score: {:5} shaped_Score: {:5} True_Avg.Score: {:.2f} shaped_Avg.Score: {:.2f}, eps-greedy: {:5.3f} Time: {:02}:{:02}:{:02}'.format(
                        i_episode, time_step, score, total_reward_shaped, avg_score, avg_score_shaped, eps, dt//3600, dt % 3600//60, dt % 60))
                if i_episode % self.hyperparams.log_every == 0 and i_episode > 0:
                    log_f.write('{},{},{},{},{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(
                        i_episode, time_step, steps_done, score, avg_score, total_reward_shaped, avg_score_shaped, eps))

            if self.training_method == "MixIn":
                scores_deque_shaped.append(total_reward_shaped)
                scores_array_shaped.append(total_reward_shaped)
                avg_score_shaped = np.mean(scores_deque_shaped)
                avg_scores_array_shaped.append(avg_score_shaped)
                if i_episode % self.hyperparams.print_every == 0 and i_episode > 0:
                    logger.debug('Episode: {:5} Timesteps: {} True_Score: {:5} shaped_Score: {:5} True_Avg.Score: {:.2f} shaped_Avg.Score: {:.2f}, eps-greedy: {:5.3f}, k: {:5.3f} Time: {:02}:{:02}:{:02}'.format(
                        i_episode, time_step, score, total_reward_shaped, avg_score, avg_score_shaped, eps, self.hyperparams.k[i_episode], dt//3600, dt % 3600//60, dt % 60))
                if i_episode % self.hyperparams.log_every == 0 and i_episode > 0:
                    log_f.write('{},{},{},{:.2f},{:.2f},{:.2f},{:5.3f},{:5.3f} \n'.format(
                        i_episode, time_step, score, avg_score, total_reward_shaped, avg_score_shaped, eps, self.hyperparams.k[i_episode]))

            if self.training_method == "epsilon_greedy" or self.training_method == "greedy":
                if i_episode % self.hyperparams.print_every == 0 and i_episode > 0:
                    logger.debug('Episode: {:5} Timesteps: {} steps_done: {} Score: {:5}  Avg.Score: {:.2f} eps-greedy: {:5.3f} Time: {:02}:{:02}:{:02}'.format(
                        i_episode, time_step, steps_done, score, avg_score, eps, dt//3600, dt % 3600//60, dt % 60))

                if i_episode % self.hyperparams.log_every == 0 and i_episode > 0:
                    log_f.write('{},{},{},{},{:.2f},{:.2f}\n'.format(
                        i_episode, time_step, steps_done, score, avg_score, eps))

            if max_100episodes_avg == None or avg_score > max_100episodes_avg:
                max_100episodes_avg = avg_score
                print("max avg_reward:", str(max_100episodes_avg))

            if max_episode_original_reward == None or score > max_episode_original_reward:
                max_episode_original_reward = score
                print("max episode reward:", str(max_episode_original_reward))

            if min_episode_original_reward == None or score < min_episode_original_reward:
                min_episode_original_reward = score
                print("min episode reward:", str(min_episode_original_reward))

            if len(scores_deque) == 100 and avg_score >= self.thereshold:
                result_logger.info('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    i_episode, np.mean(scores_deque)))
                break

            if i_episode % self.hyperparams.TARGET_UPDATE == 0:
                self.agent.q_target.load_state_dict(
                    self.agent.q_local.state_dict())

        result_logger.debug("Episode max reward : {}".format(
            max_episode_original_reward))
        result_logger.debug(
            "average of the best 100 episodes: {}".format(max_100episodes_avg))

        log_f.close()
        self.env.close()
        return run_num

    def main(self, training_method, policy):

        #get environmentt name from the hyperparameter
        self.env_name = self.hyperparams.ENV
        #get training_method and save_model from args
        self.policy = policy
        self.training_method = training_method
        self.curiosity = args.curiosity
        self.calculate_int_reward = args.cal_int_re
        save_model_ = args.save_model

        # Initialize environment
        self.env, self.thereshold, self.agent = initialize_environment(self.env_name, self.hyperparams.random_seed, self.hyperparams.num_episodes,
                                                                       self.hyperparams.BATCH_SIZE, self.hyperparams.LEARNING_RATE, self.hyperparams.GAMMA, self.hyperparams.hidden_dim, self.hyperparams.buffer_size, self.hyperparams.curiosity, self.calculate_int_reward, device)

        #start training
        run_num = self.train()

        #save model
        if save_model_ == True:
            save_model(save_model_, self.env_name,
                       self.training_method, self.agent)
        else:
            print("save_model: ", str(save_model_))

        return run_num


def init_hyperparams():
    #fixed hyperparams used for all training_methods
    hyperparams = Training.Hyperparameter
    hyperparams.ENV = 'LunarLander-v2'
    hyperparams.BATCH_SIZE = 64
    hyperparams.TAU = 0.005
    hyperparams.GAMMA = 0.99
    hyperparams.LEARNING_RATE = 0.001
    hyperparams.TARGET_UPDATE = 10
    hyperparams.buffer_size = 50000
    hyperparams.num_episodes = args.num_episodes
    hyperparams.print_every = 10
    hyperparams.log_every = 1
    hyperparams.hidden_dim = 64
    hyperparams.min_eps = 0.01
    hyperparams.max_eps_episode = 50
    hyperparams.random_seed = args.seed
    hyperparams.render = args.env_render
    hyperparams.curiosity = args.curiosity
    hyperparams.k = (np.linspace(0.5, 1.0,
                                 hyperparams.num_episodes))
    hyperparams.N = (np.linspace(1.0, 0.5,
                                 hyperparams.num_episodes))

    return hyperparams


if __name__ == '__main__':

    args = train_args()

    time_start = time.time()
    local_time = time.ctime(time_start)
    run_nums = []

    shaping_methods = ['no_shaping',
                       'GBRS', 'MixIn', 'DPBRS', 'PBRS']
    if args.shaping_method not in shaping_methods:
        raise NotImplementedError(
            f'No training method named "{args.shaping_method}"')

    if args.policy == 'epsilon_greedy' and args.shaping_method == "no_shaping":
        training_method = "epsilon_greedy"
    elif args.policy == "greedy" and args.shaping_method == "no_shaping":
        training_method = "greedy"
    else:
        training_method = args.shaping_method

    if args.curiosity != 0 and training_method != "epsilon_greedy":
        raise NotImplementedError(
            f'Intrinsic curiosity can be added only when trainig method is "epsilon_greedy". please change training method')

    for i in range(args.runs):
        result_logger.info("number of run: {} ". format(str(i)))
        hyperparameter = init_hyperparams()
        env_name = hyperparameter.ENV
        training = Training(hyperparameter)
        run_num = training.main(training_method, args.policy)
        run_nums.append(run_num)
    os.rename("logs/info.log", "logs/{}_{}_{}.log".format(
        training_method, hyperparameter.ENV, local_time))
    os.rename("results/result.log", "results/result_{}_{}_{}.log".format(
        training_method, hyperparameter.ENV, local_time))

    print(run_nums)
    if args.curiosity != 0:
        training_method_plot = "ICM"
    else:
        training_method_plot = training_method
    plot = Plot(hyperparameter.ENV, training_method_plot)
    if training_method == "epsilon_greedy" or training_method == "greedy":
        plot.plot_without_RS(local_time, run_nums[0], run_nums[-1])
    else:
        plot.plot_with_RS(local_time, run_nums[0], run_nums[-1])
