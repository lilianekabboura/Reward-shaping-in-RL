import os
import pandas as pd
import matplotlib.pyplot as plt


# smooth out rewards to get a smooth and a less smooth (var) plot lines
window_len_smooth = 20
min_window_len_smooth = 1
linewidth_smooth = 1.5
alpha_smooth = 1

window_len_var = 5
min_window_len_var = 1
linewidth_var = 2
alpha_var = 0.1

fig_num = 0

colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive',
          'brown', 'magenta', 'cyan', 'crimson', 'gray', 'black']


class Plot:

    def __init__(self, ENV, training_method):
        self.env_name = ENV
        self.training_method = training_method

    def create_directory_for_figures(self):
        # make directory for saving figures
        figures_dir = "../storage/DQN_stats_plots"
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        # make environment directory for saving figures
        figures_dir = figures_dir + '/' + self.env_name + '/'
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        # get the number of files in the directory

        return figures_dir

    def load_data(self, x, y):
        log_dir = "../storage/DQN_logs" + '/' + self.env_name + '/' + \
            self.env_name + '_' + self.training_method + '/'
        all_runs = []

        for run_num in range(x, y+1):
            log_f_name = log_dir + '/DQN_' + self.env_name + \
                "_log_" + str(run_num) + ".csv"

            print("loading data from : " + log_f_name)
            data = pd.read_csv(log_f_name)
            data = pd.DataFrame(data)

            print("data shape : ", data.shape)

            all_runs.append(data)
            print("-"*50)

        return all_runs

    def plot_without_RS(self, date, x, y):

        figures_dir = self.create_directory_for_figures()
        current_num_figs = next(os.walk(figures_dir))[2]
        fig_num = len(current_num_figs)

        fig_save_path = figures_dir + '/DQN_' + \
            self.env_name + '_stats_plot_' + \
            str(fig_num) + '_results_from_' + date + '.png'

        all_runs = self.load_data(x, y)
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        fig.suptitle('Performance of ' + self.env_name +
                     ' with ' + self.training_method)
        fig.tight_layout()

        # average all runs: true_reward
        df_concat = pd.concat(all_runs)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()
        # smooth out rewards to get a smooth and a less smooth (var) plot lines
        data_avg['reward_smooth'] = data_avg['reward'].rolling(
            window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
        data_avg['reward_var'] = data_avg['reward'].rolling(
            window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()
        data_avg.plot(kind='line', x='episode', y='reward_smooth', ax=axs[1],
                      color=colors[0],  linewidth=linewidth_smooth, alpha=alpha_smooth)
        data_avg.plot(kind='line', x='episode', y='reward_var', ax=axs[1],
                      color=colors[0],  linewidth=linewidth_var, alpha=alpha_var)
        axs[1].set_title("Avg_Reward")
        axs[1].set(xlabel="")
        axs[1].get_legend().remove()

        # plot all runs together: true_reward
        for i, run in enumerate(all_runs):
            # smooth out rewards to get a smooth and a less smooth (var) plot lines
            run['reward_smooth_' + str(i)] = run['reward'].rolling(
                window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
            run['reward_var_' + str(i)] = run['reward'].rolling(
                window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

            # plot the lines
            run.plot(kind='line', x='episode', y='reward_smooth_' + str(i), ax=axs[0],
                     color=colors[i % len(colors)],  linewidth=linewidth_smooth, alpha=alpha_smooth)
            run.plot(kind='line', x='episode', y='reward_var_' + str(i), ax=axs[0],
                     color=colors[i % len(colors)],  linewidth=linewidth_var, alpha=alpha_var)
        axs[0].set_title("allruns_Reward")
        axs[0].set(xlabel="")
        axs[0].get_legend().remove()

        print("="*50)

        plt.savefig(fig_save_path)
        print("figure saved at : ", fig_save_path)

        print("="*50)

        plt.show()

    def plot_with_RS(self, date, x, y):

        figures_dir = self.create_directory_for_figures()
        current_num_figs = next(os.walk(figures_dir))[2]
        fig_num = len(current_num_figs)

        fig_save_path = figures_dir + '/DQN_' + \
            self.env_name + '_stats_plot_' + \
            str(fig_num) + '_results_from_' + date + '.png'

        all_runs = self.load_data(x, y)
        fig, axs = plt.subplots(nrows=2, ncols=2, gridspec_kw={
                                'width_ratios': [3, 1]}, figsize=(10, 6), sharex=True)
        fig.suptitle('Performance of ' + self.env_name +
                     ' with ' + self.training_method)
        fig.tight_layout()

        # average all runs: true_reward
        df_concat = pd.concat(all_runs)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()
        # smooth out rewards to get a smooth and a less smooth (var) plot lines
        data_avg['reward_smooth'] = data_avg['reward'].rolling(
            window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
        data_avg['reward_var'] = data_avg['reward'].rolling(
            window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()
        data_avg.plot(kind='line', x='episode', y='reward_smooth', ax=axs[0, 1],
                      color=colors[0],  linewidth=linewidth_smooth, alpha=alpha_smooth)
        data_avg.plot(kind='line', x='episode', y='reward_var', ax=axs[0, 1],
                      color=colors[0],  linewidth=linewidth_var, alpha=alpha_var)
        axs[0, 1].set_title("Avg_TrueReward")
        axs[0, 1].get_legend().remove()

        # plot all runs together: true_reward
        for i, run in enumerate(all_runs):
            # smooth out rewards to get a smooth and a less smooth (var) plot lines
            run['reward_smooth_' + str(i)] = run['reward'].rolling(
                window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
            run['reward_var_' + str(i)] = run['reward'].rolling(
                window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

            # plot the lines
            run.plot(kind='line', x='episode', y='reward_smooth_' + str(i), ax=axs[0, 0],
                     color=colors[i % len(colors)],  linewidth=linewidth_smooth, alpha=alpha_smooth)
            run.plot(kind='line', x='episode', y='reward_var_' + str(i), ax=axs[0, 0],
                     color=colors[i % len(colors)],  linewidth=linewidth_var, alpha=alpha_var)
        axs[0, 0].set_title("allruns_TrueReward")
        axs[0, 0].get_legend().remove()

        # smooth out rewards to get a smooth and a less smooth (var) plot lines
        data_avg['reward_smooth_shaped'] = data_avg['shaped_reward'].rolling(
            window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
        data_avg['reward_var_shaped'] = data_avg['shaped_reward'].rolling(
            window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()
        data_avg.plot(kind='line', x='episode', y='reward_smooth_shaped', ax=axs[1, 1],
                      color=colors[0],  linewidth=linewidth_smooth, alpha=alpha_smooth)
        data_avg.plot(kind='line', x='episode', y='reward_var_shaped', ax=axs[1, 1],
                      color=colors[0],  linewidth=linewidth_var, alpha=alpha_var)
        axs[1, 1].set_title("Avg_ShapedReward")
        axs[1, 1].set(xlabel="")
        axs[1, 1].get_legend().remove()

        # plot all runs together
        for i, run in enumerate(all_runs):
            # smooth out rewards to get a smooth and a less smooth (var) plot lines
            run['reward_smooth_shaped' + str(i)] = run['shaped_reward'].rolling(
                window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
            run['reward_var_shaped' + str(i)] = run['shaped_reward'].rolling(
                window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

            # plot the lines
            run.plot(kind='line', x='episode', y='reward_smooth_shaped' + str(i), ax=axs[1, 0],
                     color=colors[i % len(colors)],  linewidth=linewidth_smooth, alpha=alpha_smooth)
            run.plot(kind='line', x='episode', y='reward_var_shaped' + str(i), ax=axs[1, 0],
                     color=colors[i % len(colors)],  linewidth=linewidth_var, alpha=alpha_var)
        axs[1, 0].set_title("allruns_ShapedReward")
        axs[1, 0].set(xlabel="")
        axs[1, 0].get_legend().remove()

        print("="*50)

        plt.savefig(fig_save_path)
        print("figure saved at : ", fig_save_path)

        print("="*50)

        plt.show()
