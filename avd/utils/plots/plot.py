import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline

sns.set()
plt.rcParams['font.family'] = 'Times New Roman'

window_size = 100
# seeds selection
seeds = ['555']
categories = ['half', 'old', 'new']
data_new = []
data_half = []
data_old = []
file_location = None

for category in ['new', 'half', 'old']:
    for seed in seeds:
        data = pd.read_csv(
            f'{file_location}/{seed}/{category}/Swimmer-v2/monitor.csv',
            skiprows=1)

        data['l'] = data['l'].cumsum()
        data.drop(columns=['t'], inplace=True)
        if category == 'new':
            data_new.append(data)
        elif category == 'half':
            data_half.append(data)
        elif category == 'old':
            data_old.append(data)

for category in ['new', 'half', 'old']:
    if category == 'new':
        temp = data_new
    elif category == 'half':
        temp = data_half
    elif category == 'old':
        temp = data_old
    episode_list = []
    reward_list = []
    for i in temp:
        reward_list.append(i['r'].values)
        episode_list.append(i['l'].values)

    final_episode_list = []
    final_reward_list = []
    for i in range(len(episode_list)):
        spline = make_interp_spline(episode_list[i], reward_list[i], k=1)
        x_smooth = np.linspace(0, 1e6, 999)

        y_smooth = spline(x_smooth)

        smoothed_series = pd.Series(y_smooth)
        smoothed_rolling_mean = smoothed_series.rolling(window=window_size).mean()
        final_reward_list.append(smoothed_rolling_mean)
        final_episode_list.append(x_smooth)
    x = np.concatenate(final_episode_list)
    y = np.concatenate(final_reward_list)
    sns.set(style="whitegrid")
    sns.lineplot(x=x, y=y)

plt.xlim([0, 1e6])
plt.show()
