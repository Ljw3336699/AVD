import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
plt.rcParams['font.family'] = 'Times New Roman'

window_size = 100
seeds = ['555']
categories = ['half', 'old', 'new']


def calculate_mean_and_std_dev(numbers):
    # mean
    mean = np.mean(numbers)

    # std
    std_dev = np.std(numbers)

    return mean, std_dev


file_location = None
data = {}
for category in categories:
    data[category] = []
    for seed in seeds:
        filepath = f'{file_location}/{seed}/{category}/Swimmer-v2/monitor.csv'
        df = pd.read_csv(filepath, skiprows=1)
        df['l'] = df['l'].cumsum()
        df['r'] = df['r'].rolling(window=window_size).mean()
        df.drop(columns=['t'], inplace=True)
        data[category].append(df)

for category in categories:
    max_values = [df['r'].max() for df in data[category]]
    mean, std_dev = calculate_mean_and_std_dev(max_values)
    print(f"Category: {category}")
    print(f"mea: {mean:.0f}")
    print(f"std: {std_dev:.0f}")
