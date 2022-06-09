import os
from logger_configs.loggers import result_logger, logger


class logging:

    def create_log_files(env_name, trainig_method):
        "Create log files (.csv) for the selected training method"

        log_dir = "../storage/DQN_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_dir = log_dir + '/' + env_name + '_' + trainig_method + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)
        # create new log file for each run
        log_f_name = log_dir + '/DQN_' + env_name + \
            "_log_" + str(run_num) + ".csv"

        result_logger.info(
            "current logging run number for " + env_name + " : {} ".format(run_num))
        result_logger.info("logging at : {} ".format(log_f_name))
        print("="*50)

        return log_f_name, run_num

    def create_log_files_episode_details(env_name):
        "Create log files (.csv) for the selected training method to log episode details"

        log_dir = "../storage/DQN_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)
        # create new log file for each run
        log_f_name = log_dir + '/DQN_' + env_name + \
            "_episode_log_" + str(run_num) + ".csv"

        result_logger.info(
            "current logging run number for " + env_name + " : {} ".format(run_num))
        result_logger.info("logging at : {} ".format(log_f_name))
        print("="*50)

        return log_f_name

    def create_log_files_potentials_details(env_name):
        "Create log files (.csv) for PBRS to log potential values details"

        log_dir = "../storage/DQN_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)
        # create new log file for each run
        log_f_name = log_dir + '/DQN_' + env_name + \
            "_potentials_log_" + str(run_num) + ".csv"

        result_logger.info(
            "current logging run number for " + env_name + " : {} ".format(run_num))
        result_logger.info("logging at : {} ".format(log_f_name))
        print("="*50)
        return log_f_name

    def create_log_files_states_details(env_name, training_method):
        "Create log files (.csv) for the selected training method to log states"

        log_dir = "../storage/DQN_episode_deep_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '_' + training_method + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)
        # create new log file for each run
        log_f_name = log_dir + '/DQN_' + env_name + \
            "_episode_log_" + str(run_num) + ".csv"

        result_logger.info(
            "current logging run number for " + env_name + " : {} ".format(run_num))
        result_logger.info(
            "logging states details at : {} ".format(log_f_name))
        print("="*50)

        return log_f_name

    def create_log_files_intrinsic_reward(env_name, training_method):
        "Create log files (.csv) for the selected training method to log intrinsic rewards"

        log_dir = "../storage/DQN_episode_intrinsic_rewards"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '_' + training_method + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)
        # create new log file for each run
        log_f_name = log_dir + '/DQN_' + env_name + \
            "_inrinsic_log_" + str(run_num) + ".csv"

        result_logger.info(
            "current logging run number for " + env_name + " : {} ".format(run_num))
        result_logger.info(
            "logging intrinsic rewards at : {} ".format(log_f_name))
        print("="*50)

        return log_f_name

########################## Checkpoint_path ############################

    def create_checkpoint_path(env_name, traning_method):
        "Create checkpoint path (.pth) for the selected training method to save model"

        directory = "../storage/DQN_preTrained"

        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + env_name + '/'
        directory = directory + '/' + env_name + '_' + traning_method + '/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        current_num_files = next(os.walk(directory))[2]
        run_num_pretrained = len(current_num_files)//2

        checkpoint_path_local = directory + \
            "DQN_{}_local_{}.pth".format(env_name, run_num_pretrained)
        checkpoint_path_target = directory + \
            "DQN_{}_target_{}.pth".format(env_name, run_num_pretrained)
        print("="*50)
        result_logger.info("save checkpoint path : {}, {}".format(
            checkpoint_path_local, checkpoint_path_target))

        return checkpoint_path_local, checkpoint_path_target
