import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

fig_width = 10
fig_height = 6

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

    def __init__(self, ENV, average, training_method, plot_all_logs):
        self.env_name = ENV
        self.plot_avg = average
        self.training_method = training_method
        self.plot_all_logs = plot_all_logs

    def create_directory_for_figures(self):
        # make directory for saving figures
        figures_dir = "../scripts/storage/DQN_plots"
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        # make environment directory for saving figures
        figures_dir = figures_dir + '/' + self.env_name + '/'
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        # get the number of files in the directory

        return figures_dir

    def load_data(self, plot_all_logs):
        log_dir = "../scripts/storage/DQN_logs" + '/' + self.env_name + \
            '/' + self.env_name + '_' + self.training_method + '/'
        current_num_files = next(os.walk(log_dir))[2]
        all_runs = []
        if plot_all_logs == True:
            num_runs = len(current_num_files)
            for run_num in range(num_runs):
                log_f_name = log_dir + '/DQN_' + self.env_name + \
                    "_log_" + str(run_num) + ".csv"
                #"_log_" + str(run_num) + "_eps=0"+ ".csv"

                print("loading data from : " + log_f_name)
                data = pd.read_csv(log_f_name)
                data = pd.DataFrame(data)

                print("data shape : ", data.shape)

                all_runs.append(data)
                print("-"*100)
        else:
            x = args.from_
            y = args.to_
            for run_num in range(x, y+1):
                log_f_name = log_dir + '/DQN_' + self.env_name + \
                    "_log_" + str(run_num) + ".csv"
        #"_log_" + str(run_num) + "_eps=0"+ ".csv"

                print("loading data from : " + log_f_name)
                data = pd.read_csv(log_f_name)
                data = pd.DataFrame(data)

                print("data shape : ", data.shape)

                all_runs.append(data)
                print("-"*100)

        return all_runs

    def create_average_or_allruns_plot(self):

        figures_dir = self.create_directory_for_figures()
        current_num_figs = next(os.walk(figures_dir))[2]
        fig_num = len(current_num_figs)

        fig_save_path = figures_dir + '/DQN_' + \
            self.env_name + '_plot_' + str(fig_num) + '_run_' + '.png'

        all_runs = self.load_data(args.plot_all_logs)
        #print(all_runs)
        ax = plt.gca()
        if self.plot_avg == True:
            # average all runs
            df_concat = pd.concat(all_runs)
            df_concat_groupby = df_concat.groupby(df_concat.index)
            data_avg = df_concat_groupby.mean()
            print(data_avg["reward"])

            # smooth out rewards to get a smooth and a less smooth (var) plot lines
            data_avg['reward_smooth'] = data_avg['reward'].rolling(
                window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
            data_avg['reward_var'] = data_avg['reward'].rolling(
                window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

            data_avg.plot(kind='line', x='episode', y='reward_smooth', ax=ax,
                          color=colors[0],  linewidth=linewidth_smooth, alpha=alpha_smooth)
            data_avg.plot(kind='line', x='episode', y='reward_var', ax=ax,
                          color=colors[0],  linewidth=linewidth_var, alpha=alpha_var)

            # keep only reward_smooth in the legend and rename it
            handles, labels = ax.get_legend_handles_labels()
            ax.legend([handles[0]], ["reward_avg_" +
                                     str(len(all_runs)) + "_runs"], loc=2)

        else:
            for i, run in enumerate(all_runs):
                # smooth out rewards to get a smooth and a less smooth (var) plot lines
                run['reward_smooth_' + str(i)] = run['avg_reward'].rolling(
                    window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
                #run['reward_var_' + str(i)] = run['reward'].rolling(
                #window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()
                #run['Avg on 100 Episode'] = run ['avg_reward'].rolling(
                #window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()

                # plot the lines
                run.plot(kind='line', x='episode', y='reward_smooth_' + str(i), ax=ax,
                         color=colors[i % len(colors)],  linewidth=linewidth_smooth, alpha=alpha_smooth)
                #run.plot(kind='line', x='episode', y='reward_var_' + str(i), ax=ax,
                #color=colors[i % len(colors)],  linewidth=linewidth_var, alpha=alpha_var)
                #run.plot(kind='line', x='episode', y='Avg on 100 Episode' , ax=ax,
                #color='blue',  linewidth=linewidth_smooth, alpha=alpha_smooth)

            # keep alternate elements (reward_smooth_i) in the legend
            handles, labels = ax.get_legend_handles_labels()
            new_handles = []
            new_labels = []
            for i in range(len(handles)):
                if(i % 2 == 0):
                    new_handles.append(handles[i])
                    new_labels.append(labels[i])
            ax.legend(new_handles, new_labels, loc=0)
            ax.get_legend().remove()

        # ax.set_yticks(np.arange(0, 1800, 200))
        # ax.set_xticks(np.arange(0, int(4e6), int(5e5)))

        ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("100 Episodes Mean Reward", fontsize=12)

        plt.title("{}, {}".format(self.env_name,
                  self.training_method), fontsize=14)

        fig = plt.gcf()
        fig.set_size_inches(fig_width, fig_height)

        print("="*100)

        plt.savefig(fig_save_path)
        print("figure saved at : ", fig_save_path)

        print("="*100)

        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str,
                        default="Acrobot-v1")
    parser.add_argument("--training_method", type=str,
                        default="epsilon_greedy")
    parser.add_argument("--plot_avg", default=False,
                        metavar='bool', type=bool)
    parser.add_argument("--plot_all_logs", default=False,
                        metavar='bool', type=bool)
    #parser.add_argument("--range_of_logs", default=False,
    #metavar='bool', type=bool)

    parser.add_argument("--from_", type=int,
                        default=0)
    parser.add_argument("--to_", type=int,
                        default=1)

    args = parser.parse_args()

    #python plot.py --env_name LunarLander-v2 --training_method MixIn --plot_all_logs True
    #python plot.py --env_name MountainCar-v0  --training_method PBRS --from_ 0 --to_ 19
    plot = Plot(args.env_name, args.plot_avg,
                args.training_method, args.plot_all_logs)
    plot.create_average_or_allruns_plot()

