#垂直堆叠条形图

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

data = pd.read_excel(os.getcwd() + '\cargo_data.xls')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.bar(np.arange(8), data.loc[0,:][1:], color = 'red', alpha = 0.8, label = '铁路', align = 'center')
plt.bar(np.arange(8), data.loc[1,:][1:], bottom = data.loc[0,:][1:], color = 'green', alpha = 0.8, label = '公路', align = 'center')
plt.bar(np.arange(8), data.loc[2,:][1:], bottom = data.loc[0,:][1:] + data.loc[1,:][1:], color = 'm', alpha = 0.8, label = '水运', align = 'center')
plt.bar(np.arange(8), data.loc[3,:][1:], bottom = data.loc[0,:][1:] + data.loc[1,:][1:] + data.loc[2,:][1:], color = 'black', alpha = 0.8, label = '民航', align = 'center')

plt.xlabel('月份')
plt.ylabel('货物量(万吨)')

plt.title('2017年各月份物流运输量')

plt.xticks(np.arange(8), data.columns[1:])
plt.ylim([10000, 500000])

for x_t, y_t in enumerate(data.loc[0,:][1:]):
	plt.text(x_t, y_t / 2, '%sW' %(round(y_t / 10000, 2)), ha = 'center', color = 'white')

for x_g, y_g in enumerate(data.loc[0,:][1:] + data.loc[1,:][1:]):
	plt.text(x_g, y_g / 2, '%sW' %(round(y_g / 10000, 2)), ha = 'center', color = 'white')

for x_s, y_s in enumerate(data.loc[0,:][1:] + data.loc[1,:][1:] + data.loc[2,:][1:]):
	plt.text(x_s, y_s-30000, '%sW' %(round(y_s/10000, 2)), ha = 'center', color = 'white')

for x_f, y_f in enumerate(data.loc[0,:][1:] + data.loc[1,:][1:] + data.loc[2,:][1:] + data.loc[3,:][1:]):
	plt.text(x_f, y_f-10000, '%sW' %(round(y_f/10000, 2)), ha = 'center', color = 'white')

plt.legend(loc = 'upper center', ncol = 4)

plt.show()