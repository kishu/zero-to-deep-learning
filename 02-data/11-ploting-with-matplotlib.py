import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# numpy.random.normal(loc=0.0, scale=1.0, size=None)
# Draw random samples from a normal (Gaussian) distribution.
data1 = np.random.normal(0, 0.1, 1000)

# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)[source]
# Return evenly spaced numbers over a specified interval.
data2 = np.random.normal(1, 0.4, 1000) + np.linspace(0, 1, 1000)
data3 = 2 + np.random.random(1000) * np.linspace(1, 5, 1000)
data4 = np.random.normal(3, 0.2, 1000) + 0.3 * np.sin(np.linspace(0, 20, 1000))

data = np.vstack([data1, data2, data3, data4]).transpose()
df = pd.DataFrame(data, columns=['data1', 'data2', 'data3', 'datat4'])

#
# Line Plot
#
# df.plot(title='Line plot')
# plt.plot(df)
# plt.title('Line plot')
# plt.legend(['data1', 'data2', 'data3', 'data4'])
# plt.show()

#
# Scatter Plot
#
# df.plot(style='.')
# _ =  df.plot(kind='scatter', x='data1', y='data2', xlim=(-1.5, 1.5), ylim=(0, 3))
# plt.show()

#
# Histogram
#
# df.plot(kind='hist',
#         bins=50,
#         title='Histogram',
#         alpha=0.6,
#         figsize=(12, 8))
# plt.show()

#
# Cumulative distribution
# 
# df.plot(kind='hist',
#     bins=100,
#     title='Cumulative distributions',
#     normed=True,
#     cumulative=True,
#     alpha=0.4,
#     figsize=(12, 8))
# plt.show()

#
# Box Plot
#
# df.plot(kind='box',
#         title='Boxplot')
# plt.show()

#
# Subplots
#
# fig, ax = plt.subplots(2, 2, figsize=(5, 5))
# df.plot(ax=ax[0][0],
#         title='Line plot')
# df.plot(ax=ax[0][1],
#     style='o',
#     title='Scatter plot')
# df.plot(ax=ax[1][0],
#     kind='hist',
#     title='Histogram')
# df.plot(ax=ax[1][1],
#     kind='box',
#     title='Boxplot')
# tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area.
# plt.tight_layout()
# plt.show()

#
# Pie charts
#
# gt01 = df['data1'] > 0.1
# piecounts = gt01.value_counts()
# piecounts.plot(kind='pie',
#                 figsize=(5, 5),
#                 explode=[0, 0.15],
#                 labels=['<= 0.1', '> 0.1'],
#                 autopct='%1.1f%%',
#                 shadow=True,
#                 startangle=90,
#                 fontsize=16)
# plt.show()

#
# Hexbin plot
#
data = np.vstack([np.random.normal((0, 0), 2, size=(1000, 2)),
                np.random.normal((0, 0), 3, size=(2000, 2))])
df = pd.DataFrame(data, columns=['x', 'y'])
df.plot()
df.plot(kind='hexbin', x='x', y='y', bins=100, cmap='rainbow')
plt.show()