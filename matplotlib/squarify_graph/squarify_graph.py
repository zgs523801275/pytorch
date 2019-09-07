#树形图

import matplotlib.pyplot as plt
import squarify

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

name = ['国内增值税', '国内消费税', '企业所得税', '个人所得税',
  '进口增值税/消费税', '出口退税', '城市维护建设税', '车辆购置税',
  '印花税', '资源税', '土地/房地产相关税', '环境保护税', '车船/船舶吨/烟叶税']

income = [41017, 9414, 30369, 6433, 9548, 10736, 3029, 2142, 1597, 1135, 11690, 165, 596]

colors = ['#ff9999', '#9999ff', '#aa5555', '#5555aa', '#dd3333', '#3333dd']

plt.figure(figsize = (9, 5))

plot = squarify.plot(sizes = income,
	label = name,
	color = colors,
	alpha = 0.6,
	value = income,
	edgecolor = 'white',
	linewidth = 3
	)

plt.rc('font', size = 8)

plot.set_title('2019年7月财政收支情况(亿元)', fontdict = {'fontsize': 15})

plt.axis('off')

plt.tick_params(top = False, right = False)

plt.show()