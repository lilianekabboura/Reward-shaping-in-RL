
import os
import pandas as pd

env_name = 'Acrobot-v1'
#env_name = 'MountainCar-v0'
#env_name = 'LunarLander-v2'
trainig_method= "_ICM"
datalist = []
#log_dir = "../scripts/storage/DQN_episode_logs" + '/' + env_name + '/'
log_dir = "../scripts/storage/DQN_logs" + '/' + env_name + '/'+ env_name + trainig_method

for count, file in enumerate(os.listdir(log_dir)):
           print(file)
           data_for_boxplot = []
           data = pd.read_csv(os.path.join(log_dir, file))
           data = pd.DataFrame(data)
           
           data = data['avg_reward'].tolist()
           for value in data:
               datalist.append(value)
#print (data)
#m = min(i for i in datalist if i !=-100)
m = min(i for i in datalist)

print(m)
#max = max(i for i in datalist if (i !=100 and i < 100))
max = max(i for i in datalist)
print(max)
