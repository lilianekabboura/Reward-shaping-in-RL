
import os
import pandas as pd
from logger_configs.loggers_states import result_logger_2
import time
from pprint import pformat
from statistics import mean, stdev


def create_log_files_episode_details_for_stats(env_name, training_method):

    log_dir = "../storage/DQN_episode_states_for_stats"
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
        "_episode_log_states_" + str(run_num) + ".csv"

    result_logger_2.info(
        "current logging run number for " + env_name + " : {} ".format(run_num))
    result_logger_2.info("logging at : {} ".format(log_f_name))
    print("="*50)

    return log_f_name


def states(env_name, training_method, run_num):
    datalist = []

    log_dir = "../storage/DQN_episode_deep_logs" + '/' + env_name + '//'+env_name+'_' + training_method + '//' + 'DQN_' + env_name + \
        "_episode_log_" + str(run_num) + ".csv"
    time_start = time.time()
    local_time = time.ctime(time_start)
    data = pd.read_csv(os.path.join(log_dir))
    run = []
    data = pd.DataFrame(data)
    run.append(data)

    data = data['state'].tolist()
    for value in data:
        value = value.strip('][').split(' ')
        datalist.append(value)

    data_2 = []
    for st in datalist:
        st = [x for x in st if x]
        data_2.append(st)
    data_3 = []
    for v in data_2:
        v = [float(x) for x in v]
        data_3.append(v)
    x = print(*map(mean, zip(*data_3)))
    print(*map(stdev, zip(*data_3)))

    result_logger_2.info("=" * 110)
    result_logger_2.info("OpenAI Gym Task : {} ".format(env_name))
    result_logger_2.info("Training_method : {} ".format(training_method))
    result_logger_2.info("loading data from : {} ".format(log_dir))

    result_logger_2.info(
        "count of all visited states during trainig : {}".format(len(data)))
    result_logger_2.info("average of the visited states during trainig: {} {}".format(
        *map(mean, zip(*data_3))))
    result_logger_2.info("standart deviation of the visited states during trainig: {} {}".format(
        *map(stdev, zip(*data_3))))

    my_dict = {i: data.count(i) for i in data}

    data_after_count = my_dict
    result_logger_2.info("count of visited states during training after mapping them together : {} ".format(
        len(data_after_count)))
    result_logger_2.info("all visited states : {} ".format(pformat(my_dict)))
    for key, value in list(data_after_count.items()):
        if value == 1:
            del data_after_count[key]
    result_logger_2.info("states visited more than once : {} ".format(
        pformat(data_after_count)))
    result_logger_2.info("count of states visited more than once : {} ".format(
        pformat(len(data_after_count))))

    new_data_list = data_after_count

    log_f_name = create_log_files_episode_details_for_stats(
        env_name, training_method)
    log_f = open(log_f_name, "w+")
    log_f.write('state,count\n')
    data_4 = []
    for key, value in list(new_data_list.items()):
        log_f.write('{},{}\n'.format(key, value))
        data_4.append(key)

    data_5 = []
    for value in data_4:
        value = value.strip('][').split(' ')
        data_5.append(value)

    data_6 = []
    for st in data_5:
        st = [x for x in st if x]
        data_6.append(st)

    data_7 = []
    for v in data_6:
        v = [float(x) for x in v]
        data_7.append(v)

    print(*map(mean, zip(*data_7)))

    result_logger_2.info("average of the visited states more than once: {} {}".format(
        *map(mean, zip(*data_7))))
    result_logger_2.info("standart deviation of the visited states during trainig: {} {}".format(
        *map(stdev, zip(*data_7))))

    #log_f.close()
    os.rename("results_states/result_states_logs.log", "results_states/result_states_log_{}_{}.log".format(
        env_name, local_time))


if __name__ == '__main__':

    envs = ['Acrobot-v1', 'MountainCar-v0', 'LunarLander-v2']
    training_method = ['epsilon_greedy', 'greedy',
                       'GBRS', 'MixIn', 'DPBRS', 'PBRS', 'ICM']
    states(envs[1], training_method[0], 2)
