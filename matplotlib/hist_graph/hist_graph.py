#直方图

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

titanic = pd.read_csv('titanic_train.csv')

any(titanic.Age.isnull())
titanic.dropna(subset = ['Age'], inplace = True)

plt.style.use('ggplot')

plt.hist(titanic.Age,
	bins = 20,
	color = 'steelblue',
	edgecolor = 'k',
	label = '直方图'
	)

plt.tick_params(top = False, right = False)

plt.legend()

plt.show()

plt.hist(titanic.Age,
	bins = np.arange(titanic.Age.min(), titanic.Age.max(), 5),
	density = True,
	cumulative = True,
	color = 'steelblue',
	edgecolor = 'k',
	label = '直方图'
	)

plt.title('乘客年龄的频率累计直方图')
plt.xlabel('年龄')
plt.ylabel('累计频率')

plt.tick_params(top = False, right = False)

plt.legend(loc = 'best')

plt.show()

plt.hist(titanic.Age,
	bins = np.arange(titanic.Age.min(), titanic.Age.max(), 5),
	density = True,
	color = 'steelblue',
	edgecolor = 'k'
	)

plt.title('乘客年龄直方图')
plt.xlabel('年龄')
plt.ylabel('频率')

x1 = np.linspace(titanic.Age.min(), titanic.Age.max(), 1000)
normal = mlab.normpdf(x1, titanic.Age.mean(), titanic.Age.std())

line1, = plt.plot(x1, normal, 'r-', linewidth = 2)

kde = mlab.GaussianKDE(titanic.Age)
x2 = np.linspace(titanic.Age.min(), titanic.Age.max(), 1000)

line2, = plt.plot(x2, kde(x2), 'g-', linewidth = 2)

plt.tick_params(top = False, right = False)

plt.legend([line1, line2], ['正态分布曲线', '核密度曲线'], loc = 'best')

plt.show()

age_female = titanic.Age[titanic.Sex == 'female']
age_male = titanic.Age[titanic.Sex == 'male']

bins = np.arange(titanic.Age.min(), titanic.Age.max(), 2)

plt.hist(age_male, 
	bins = bins, 
	label = '男性', 
	color = 'steelblue',
	alpha = 0.7
	)

plt.hist(age_female, 
	bins = bins, 
	label = '女性', 
	alpha = 0.7
	)

plt.title('乘客年龄直方图')
plt.xlabel('年龄')
plt.ylabel('人数')

plt.tick_params(top = False, right = False)

plt.legend()

plt.show()