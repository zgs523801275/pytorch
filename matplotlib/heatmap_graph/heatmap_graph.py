#热力图

import datetime
import calendar
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

headers = {
	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
}

url = 'http://lishi.tianqi.com/zhengzhou/201908.html'

response = requests.get(url, headers = headers).text

soup = BeautifulSoup(response, 'html.parser')

datas = soup.findAll('div', {'class': 'tqtongji2'})[0].findAll('ul')[1:]

date = [i.findAll('li')[0].text for i in datas]
high = [i.findAll('li')[1].text for i in datas]

df = pd.DataFrame({'date': date, 'high': high})


df.date = pd.to_datetime(df.date)

df.high = df.high.astype('int')

df['weekday'] = df.date.apply(pd.datetime.weekday)

def week_of_month(tgtdate):
	days_this_month = calendar.mdays[tgtdate.month]
	for i in range(1, days_this_month + 1):
		d = datetime.datetime(tgtdate.year, tgtdate.month, i)
		if d.day - d.weekday() > 0:
			startdate = d
			break
	return (tgtdate - startdate).days // 7 + 1

df['week_of_month'] = df.date.apply(week_of_month)


target = pd.pivot_table(data = df.iloc[:, 1:], values = 'high',
	index = 'week_of_month', columns = 'weekday')

target.fillna(0, inplace = True)

target.index.name = None

target.sort_index(ascending = False, inplace = True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.pcolor(target,
	cmap = plt.cm.Blues,
	edgecolor = 'white'
	)

plt.xticks(np.arange(7) + 0.5, ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
plt.yticks(np.arange(5) + 0.5, ['第五周', '第四周', '第三周', '第二周', '第一周'])

plt.tick_params(top = False, right = False)

plt.title('郑州市2019年8月份每日最高气温分布图')

plt.show()
#--------------------------------------------------------------------------
target = pd.pivot_table(data = df.iloc[:, 1:], values = 'high',
	index = 'week_of_month', columns = 'weekday')

ax = sns.heatmap(target,
	cmap = plt.cm.Blues,
	linewidth = 1,
	annot = True
	)

plt.xticks(np.arange(7) + 0.5, ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])

plt.yticks(np.arange(5) + 0.5, ['第一周', '第二周', '第三周', '第四周', '第五周'], rotation = 0)

ax.set_title('郑州市2019年8月份每日最高气温分布图')
ax.set_xlabel('')
ax.set_ylabel('')

plt.show()