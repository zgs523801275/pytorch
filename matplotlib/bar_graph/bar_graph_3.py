#水平交错条形图

import matplotlib.pyplot as plt
import numpy as np

Y2016 = [15600, 12700, 11300, 4230, 3620]
Y2017 = [17400, 14800, 12000, 5200, 4020]
labels = ['北京', '上海', '香港', '深圳', '广州']
bar_width = 0.45

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.bar(np.arange(5), Y2016, label = '2016', color = 'steelblue', alpha = 0.8, width = bar_width)
plt.bar(np.arange(5) + bar_width, Y2017, label = '2017', color = 'indianred', alpha = 0.8, width = bar_width)

plt.xlabel('Top5城市')
plt.ylabel('家庭数量')

plt.title('亿万财富家庭数Top5城市分布')

plt.xticks(np.arange(5) + bar_width/2, labels, ha = 'center')
plt.ylim([2500, 19000])

for x2016, y2016 in enumerate(Y2016):
	plt.text(x2016, y2016 + 100, '%s' %y2016, ha = 'center')

for x2017, y2017 in enumerate(Y2017):
	plt.text(x2017 + bar_width, y2017 + 100, '%s' %y2017, ha = 'center')

plt.legend()

plt.show()