#散点图

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

cars = pd.read_csv('cars.csv')

plt.scatter(cars.speed,
	cars.dist,
	s = 30,
	c = 'steelblue',
	marker = 's',
	alpha = 0.9,
	linewidths = 0.3,
	edgecolor = 'red'
	)

plt.title('汽车速度与刹车距离的关系')
plt.xlabel('汽车速度')
plt.ylabel('刹车距离')

plt.tick_params(top = False, right = False)

plt.show()
#--------------------------------------------------------------
iris = pd.read_csv('iris.csv')

colors = ['steelblue', '#9999ff', '#ff9999']

species = iris.Species.unique()

for i in range(len(species)):
	plt.scatter(iris.loc[iris.Species == species[i], 'Petal.Length'],
		iris.loc[iris.Species == species[i], 'Petal.Width'],
		s =35,
		c = colors[i],
		label = species[i]
		)

plt.title('花瓣长度和宽度的关系')
plt.xlabel('花瓣长度')
plt.ylabel('花瓣宽度')

plt.tick_params(top = False, right = False)

plt.legend(loc = 'upper left')

plt.show()
#-----------------------------------------------------------------------
import numpy as np

sales = pd.read_excel('sales.xlsx')

plt.scatter(sales.finish_ratio,
	sales.profit_ratio,
	c = 'steelblue',
	s = sales.tot_target / 30,
	edgecolor = 'black'
	)

plt.xticks(np.arange(0, 1, 0.1), [str(round(i * 100, 1)) + '%' for i in np.arange(0, 1, 0.1)])
plt.yticks(np.arange(0, 1, 0.1), [str(round(i * 100, 1)) + '%' for i in np.arange(0, 1, 0.1)])

plt.xlim(0.2, 0.7)
plt.ylim(0.25, 0.85)

plt.title('完成率与利润率的关系')
plt.xlabel('完成率')
plt.ylabel('利润率')

plt.tick_params(top = False, right = False)

plt.show()
#----------------------------------------------------------------------
from sklearn.linear_model import LinearRegression

plt.scatter(cars.speed,
	cars.dist,
	s = 30,
	c = 'steelblue',
	marker = 'o',
	alpha = 0.9,
	linewidths = 0.3,
	edgecolor = 'red',
	label = '观测点'
	)

reg = LinearRegression().fit(cars.speed.values.reshape(-1, 1), cars.dist)

pred = reg.predict(cars.speed.values.reshape(-1, 1))

plt.plot(cars.speed, pred, linewidth = 2, label = '回归线')

plt.title('汽车速度与刹车距离的关系')
plt.xlabel('汽车速度')
plt.ylabel('刹车距离')

plt.tick_params(top = False, right = False)

plt.legend(loc = 'upper left')

plt.show()