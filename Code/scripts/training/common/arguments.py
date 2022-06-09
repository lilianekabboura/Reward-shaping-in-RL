import argparse


def train_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str,
                        default="epsilon_greedy", help="Policy to use for training. Choose between 'epsilon_greedy' 'greedy'. (default: epsilon_greedy)")
    parser.add_argument("--shaping_method", type=str,
                        default="no_shaping", help="Include shaping method in training. Choose between 'no_shaping', 'GBRS', 'MixIn', 'DPBRS', 'PBRS'. (default: no_shaping)")
    parser.add_argument("--num_episodes", default=2000, type=int,
                        help="Number of episodes of training. (default: 2000)")
    parser.add_argument("--save_model", default=False,
                        metavar='bool', type=bool, help="Boolean to specify if the trained network shall be saved. (default: False)")
    parser.add_argument("--runs", default=1, type=int,
                        help=" Number of runs should be performed. (default:1)")
    parser.add_argument("--env_render", default=False,
                        metavar='bool', type=bool, help="Boolean to enable environment rendering during training. (default: False)")
    parser.add_argument("--seed", default=1, type=int,
                        help="andom seed to reproduce training runs (default: 1)")
    parser.add_argument("--deep_logging", default=False,
                        metavar='bool', type=bool, help="Boolean to enable deep logging of visited states during training for the purpose of statistics. (default: False)")
    parser.add_argument("--curiosity", default=0, type=int,
                        help="Adding intrinsic curiosity to the extrinsic reward. Choose between 0:no curiosity, 1:reward and curiosity, 2:only curiosity. (default: 0)")
    parser.add_argument("--cal_int_re", default=False, type=bool,
                        help="Boolean to enable intrisic reward calculation when ICM is not applied. (default: False)")

    return parser.parse_args()


def vis_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True,
                        default="Acrobot-v1", help="Name of the environment. Choose between 'Acrobot-v1' 'MountainCar-v0' 'LunarLander-v2'. (default: Acrobot-v1)")
    parser.add_argument("--training_method", type=str,
                        default="epsilon_greedy", help="Method for training. Choose between 'epsilon_greedy' 'GBRS' 'MixIn' 'DPBRS' 'PBRS' 'ICM'. (default: epsilon_greedy)")
    parser.add_argument("--num_episodes", default=10, type=int,
                        help="Maximum number of episodes to be executed. (default: 10)")
    parser.add_argument("--seed", default=1, type=int,
                        help="Generate the same set of pseudo random constellations, colors, positions, etc. every time the algorithm is executed. (default: 1)")

    parser.add_argument("--run_num", default=0, type=int,
                        help="the number of the run of the save model. (default: 0)")

    return parser.parse_args()
