#面积图

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('ggplot')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

transport = pd.read_excel('transport.xls')
print(transport.head(5))
print(transport.shape[1])
N = np.arange(transport.shape[1] - 1)

labels = transport.Index
channel = transport.columns[1:]

for i in range(transport.shape[0]):
	plt.plot(N,
		transport.loc[i, 'Jan' : 'Aug'],
		label = labels[i],
		marker = 'o',
		linewidth = 2
		)

plt.title('2017年各运输渠道的运输量')
plt.ylabel('运输量(万吨)')

plt.xticks(N, channel)

plt.tick_params(top = False, right = False)

plt.legend(loc = 'best')

plt.show()
#---------------------------------------------------------
y1 = transport.loc[0, 'Jan' : 'Aug'].astype('int')
y2 = transport.loc[1, 'Jan' : 'Aug'].astype('int')
y3 = transport.loc[2, 'Jan' : 'Aug'].astype('int')
y4 = transport.loc[3, 'Jan' : 'Aug'].astype('int')

colors = ['#ff9999', '#9999ff', '#aa5555', '#5555aa']

plt.stackplot(N,
	y1,y2,y3,y4,
	labels = labels,
	colors = colors
	)

plt.title('2017年各运输渠道的运输量')
plt.ylabel('运输量(万吨)')

plt.xticks(N, channel)

plt.tick_params(top = False, right = False)

plt.legend(loc = 'upper left')

plt.show()