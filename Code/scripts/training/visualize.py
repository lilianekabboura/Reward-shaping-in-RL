from dataclasses import dataclass
import torch
from common.DQN.agent import FloatTensor
from common.make_env import initialize_environment
import time

from collections import deque
import numpy as np
from common.arguments import vis_args


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Visualize:
    @dataclass
    class Hyperparameter:
        ENV: str
        BATCH_SIZE: int
        GAMMA: float
        LEARNING_RATE: float
        buffer_size: int
        num_episodes: int
        hidden_dim: int
        random_seed: float

    def __init__(self, hyperparams):

        self.env = None
        self.env_name = None
        self.agent = None
        self.thereshold = None
        self.policy = None
        self.training_method = None
        self.hyperparams = hyperparams

    def load(self, agent, directory, filename, run_num):

        agent.q_local.load_state_dict(torch.load(
            '%s/%s_local_%s.pth' % (directory,  filename, run_num)))
        agent.q_target.load_state_dict(torch.load(
            '%s/%s_target_%s.pth' % (directory,  filename, run_num)))

    def play(self, env, agent, n_episodes):

        scores_deque = deque(maxlen=100)
        for i_episode in range(1, n_episodes+1):
            s = env.reset()

            total_reward = 0
            time_start = time.time()
            timesteps = 0

            while True:

                a = agent.get_action(FloatTensor(
                    [s]), check_eps=False, eps=0.001)
                env.render()
                s2, r, done, _ = env.step(a.item())
                s = s2
                total_reward += r
                timesteps += 1

                if done:
                    break

            delta = (int)(time.time() - time_start)

            scores_deque.append(total_reward)

            print('Episode {}\tAverage Score: {:.2f}, \t Timesteps: {} \tTime: {:02}:{:02}:{:02}'
                  .format(i_episode, np.mean(scores_deque), timesteps,
                          delta//3600, delta % 3600//60, delta % 60))

    def main(self, training_method, run_num, n_episodes):

        #get environmentt name from the hyperparameter
        self.env_name = self.hyperparams.ENV
        #get training_method and save_model from args
        self.training_method = training_method
        directory = '../storage/DQN_pretrained' + '/' + self.env_name + '/' + \
            self.env_name + '_' + self.training_method + '/'
        file_name = 'DQN_' + self.env_name

        # Initialize environment
        self.env, self.thereshold, self.agent = initialize_environment(self.env_name, self.hyperparams.random_seed, self.hyperparams.num_episodes,
                                                                       self.hyperparams.BATCH_SIZE, self.hyperparams.LEARNING_RATE, self.hyperparams.GAMMA, self.hyperparams.hidden_dim, self.hyperparams.buffer_size, curiosity=0, int_re=False, device=device)

        self.load(self.agent, directory, file_name, run_num)
        self.play(self.env, self.agent, n_episodes)


def init_hyperparams():
    #fixed hyperparams
    hyperparams = Visualize.Hyperparameter
    hyperparams.ENV = args.env
    hyperparams.BATCH_SIZE = 64
    hyperparams.GAMMA = 0.99
    hyperparams.LEARNING_RATE = 0.001
    hyperparams.buffer_size = 50000
    hyperparams.num_episodes = args.num_episodes
    hyperparams.hidden_dim = 64  # 16 #if env is Acrobot hidden_dim should be 16
    hyperparams.random_seed = args.seed

    return hyperparams


if __name__ == '__main__':
    args = vis_args()

    hyperparameter = init_hyperparams()
    env_name = hyperparameter.ENV
    visualize = Visualize(hyperparameter)
    visualize.main(args.training_method, args.run_num, args.num_episodes)
