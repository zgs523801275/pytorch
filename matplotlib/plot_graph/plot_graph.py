#折线图

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

article_reading = pd.read_excel('wechart.xlsx')

sub_data = article_reading.loc[article_reading.date >= '2017-08-01', :]
#------------------------------------------------------------------------
fig = plt.figure(figsize = (8, 5))

plt.plot(sub_data.date,
	sub_data.article_reading_cnts,
	linestyle = '-',
	linewidth = 2,
	color = 'steelblue',
	marker = 'o',
	markersize = 6,
	markeredgecolor = 'black',
	markerfacecolor = 'brown'
	)

plt.title('公众号每天阅读人数趋势图')
plt.xlabel('日期')
plt.ylabel('人数')

plt.tick_params(top = False, right = False)

fig.autofmt_xdate(rotation = 45)

plt.show()
#------------------------------------------------------------------
import matplotlib as mpl

fig = plt.figure(figsize = (8, 5))

plt.plot(sub_data.date,
	sub_data.article_reading_cnts,
	linestyle = '-',
	linewidth = 2,
	color = 'steelblue',
	marker = 'o',
	markersize = 6,
	markeredgecolor = 'black',
	markerfacecolor = 'brown'
	)

plt.title('公众号每天阅读人数趋势图')
plt.xlabel('日期')
plt.ylabel('人数')

plt.tick_params(top = False, right = False)

ax = plt.gca()

date_format = mpl.dates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_format)

#xlocator = mpl.ticker.LinearLocator(10)

xlocator = mpl.ticker.MultipleLocator(5)
ax.xaxis.set_major_locator(xlocator)

fig.autofmt_xdate(rotation = 45)

plt.show()
#------------------------------------------------------------------
fig = plt.figure(figsize = (8, 5))

plt.plot(sub_data.date,
	sub_data.article_reading_cnts,
	linestyle = '-',
	linewidth = 2,
	color = 'steelblue',
	marker = 'o',
	markersize = 6,
	markeredgecolor = 'black',
	markerfacecolor = 'steelblue',
	label = '阅读人数'
	)

plt.plot(sub_data.date,
	sub_data.article_reading_times,
	linestyle = '-',
	linewidth = 2,
	color = '#ff9999',
	marker = 'o',
	markersize = 6,
	markeredgecolor = 'black',
	markerfacecolor = '#ff9999',
	label = '阅读人次'
	)

plt.title('公众号每天阅读人数趋势图')
plt.xlabel('日期')
plt.ylabel('人数')

plt.tick_params(top = False, right = False)

ax = plt.gca()

date_format = mpl.dates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_format)

#xlocator = mpl.ticker.LinearLocator(10)

xlocator = mpl.ticker.MultipleLocator(5)
ax.xaxis.set_major_locator(xlocator)

fig.autofmt_xdate(rotation = 45)

plt.show()