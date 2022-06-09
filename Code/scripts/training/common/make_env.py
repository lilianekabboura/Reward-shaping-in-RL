import gym
from logger_configs.loggers import result_logger, logger
from .DQN.agent import Agent


def initialize_environment(env_name, seed, num_episodes, BATCH_SIZE,  LEARNING_RATE, GAMMA, hidden_dim, buffer_size, curiosity, int_re, device):

    env = gym.make(env_name)
    env.seed(seed)
    space_dim = env.observation_space.shape  # n_spaces
    action_dim = env.action_space.n  # n_action
    result_logger.info("=" * 110)
    result_logger.info("OpenAI Gym Task : {} ".format(env_name))
    result_logger.info("Initialised with parameters")
    result_logger.debug("Parameters list : num_episodes: {}, batch_size: {}, lr: {}, gamma: {}".format(
        num_episodes, BATCH_SIZE, LEARNING_RATE, GAMMA))

    logger.debug('State shape: {}'.format(
        env.observation_space.shape))
    logger.debug('Number of actions: {}'.format(env.action_space.n))
    result_logger.debug("input_dim:{}, output_dim: {}, hidden_dim: {}".format(
        space_dim, action_dim, hidden_dim))

    print("="*50)
    print('input_dim: ', space_dim, ', output_dim: ',
          action_dim, ', hidden_dim: ', hidden_dim)

    threshold = env.spec.reward_threshold
    print('threshold: ', threshold)
    logger.debug('threshold: {}'.format(threshold))
    print("="*50)
    print("trainnig the " + env_name + " environment")
    print("="*50)

    agent = Agent(space_dim, action_dim, hidden_dim, buffer_size, curiosity,int_re, device,
                  lr=LEARNING_RATE, logger=logger)
    return env, threshold, agent
