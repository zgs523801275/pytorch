import matplotlib.pyplot as plt

#设置绘图主题风格
plt.style.use('ggplot')

edu = [0.2515, 0.3724, 0.3336, 0.0368, 0.0057]
labels = ['中专', '大专', '本科', '硕士', '其他']

#用于突出显示大专学历人群
explode = [0, 0.1, 0, 0, 0]

#自定义颜色
colors = ['#9999ff', '#ff9999', '#7777aa', '#2442aa', '#dd5555']

#中文乱码和坐标轴负号处理
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#将横纵坐标轴标准化处理，保证饼图是一个正圆，否则为椭圆
plt.axes(aspect = 'equal')

plt.xlim(0, 4)
plt.ylim(0, 4)

#绘图
plt.pie(x = edu, #绘制数据
	explode = explode, #突出显示大专人群
	labels = labels, #添加教育标签
	colors = colors, #设置饼图自定义填充色
	autopct = '%.1f%%', #设置百分比格式, 保留一位小数
	pctdistance = 0.8, #设置百分比标签与圆心的距离
	labeldistance = 1.15, #设置教育标签与圆心的距离
	startangle = 180, #设置饼图初始角度
	radius = 1.5, #设置饼图半径
	counterclock = False, #是否逆时针, 这里设置为顺时针
	wedgeprops = {'linewidth': 1.5, 'edgecolor': 'green'}, #设置饼图内外边界的属性值
	textprops = {'fontsize': 12, 'color': 'k'}, #设置文本标签的属性值
	center = (1.8, 1.8), #设置饼图的圆点
	frame = 1 #是否显示饼图的图框, 这里设置显示
	)

plt.xticks(())
plt.yticks(())

plt.title('芝麻信用失信用户教育水平分布')

plt.show()