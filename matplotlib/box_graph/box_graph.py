# 导入第三方模块
import pandas as pd
import matplotlib.pyplot as plt

# 读取Titanic数据集
titanic = pd.read_csv('titanic_train.csv')

# 检查年龄是否有缺失
any(titanic.Age.isnull())
# 不妨删除含有缺失年龄的观察
titanic.dropna(subset=['Age'], inplace=True)

# 设置图形的显示风格
plt.style.use('ggplot')

# 设置中文和负号正常显示
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 绘图：整体乘客的年龄箱线图
plt.boxplot(x = titanic.Age, # 指定绘图数据
            patch_artist = True, # 要求用自定义颜色填充盒形图，默认白色填充
            showmeans = True, # 以点的形式显示均值
            boxprops = {'color':'black','facecolor':'#9999ff'}, # 设置箱体属性，填充色和边框色
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
            meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色
            medianprops = {'linestyle':'--','color':'orange'} # 设置中位数线的属性，线的类型和颜色
            )

# 设置y轴的范围
plt.ylim(0,85)

# 去除箱线图的上边框与右边框的刻度标签
plt.tick_params(top = False, right = False)

# 显示图形
plt.show()

#按舱级排序，为了后面正常显示分组盒形图的顺序
titanic.sort_values(by = 'Pclass', inplace = True)

#将不同舱位的年龄人群分别存储到列表Age变量中
Age = []
Levels = titanic.Pclass.unique()
for Pclass in Levels:
	Age.append(titanic.loc[titanic.Pclass == Pclass, 'Age'])

plt.boxplot(x = Age, # 指定绘图数据
            patch_artist = True, # 要求用自定义颜色填充盒形图，默认白色填充
            labels = ['一等舱', '二等舱', '三等舱'], #添加具体标签名称
            showmeans = True, # 以点的形式显示均值
            boxprops = {'color':'black','facecolor':'#9999ff'}, # 设置箱体属性，填充色和边框色
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
            meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色
            medianprops = {'linestyle':'--','color':'orange'} # 设置中位数线的属性，线的类型和颜色
            )

plt.show()