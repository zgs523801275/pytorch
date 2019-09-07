#雷达图

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('ggplot')

values1 = [3.2, 2.1, 3.5, 2.8, 3]
feature = ['个人能力', 'QC知识', '解决问题能力', '服务质量意识', '团队精神']

angles = np.linspace(0, 2 * np.pi, len(values1), endpoint = False)

values1 = np.concatenate((values1, [values1[0]]))
angles = np.concatenate((angles, [angles[0]]))

fig = plt.figure()

ax = fig.add_subplot(111, polar = True)

ax.plot(angles, values1, '-o', linewidth = 2)

ax.fill(angles, values1, alpha = 0.25)

ax.set_thetagrids(angles * 180 / np.pi, feature)

ax.set_ylim(0, 5)

plt.title('活动前员工状态表现')

ax.grid(True)

plt.show()
#----------------------------------------------------------
values1 = [3.2, 2.1, 3.5, 2.8, 3]
values2 = [4, 4.1, 4.5, 4.4, 4.1]
feature = ['个人能力', 'QC知识', '解决问题能力', '服务质量意识', '团队精神']

angles = np.linspace(0, 2 * np.pi, len(values1), endpoint = False)

values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
angles = np.concatenate((angles, [angles[0]]))

fig = plt.figure()

ax = fig.add_subplot(111, polar = True)

ax.plot(angles, values1, '-o', linewidth = 2, label = '活动前')
ax.plot(angles, values2, '-o', linewidth = 2, label = '活动后')

ax.fill(angles, values1, alpha = 0.25)
ax.fill(angles, values2, alpha = 0.25)

ax.set_thetagrids(angles * 180 / np.pi, feature)

ax.set_ylim(0, 5)

plt.title('活动前后员工状态表现')

ax.grid(True)

plt.legend(loc = 'upper left')

plt.show()
#-------------------------------------------------
import pygal

radar_chart = pygal.Radar(fill = True, range = (0, 5))

radar_chart.title = '活动前后员工状态表现'

radar_chart.x_labels = ['个人能力', 'QC知识', '解决问题能力', '服务质量意识', '团队精神']

radar_chart.add('活动前', [3.2, 2.1, 3.5, 2.8, 3])
radar_chart.add('活动后', [4, 4.1, 4.5, 4.4, 4.1])

radar_chart.render_to_file('radar_chart.svg')