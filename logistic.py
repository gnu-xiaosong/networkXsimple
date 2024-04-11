import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from pylab import mpl
import random
import pandas as pd
from time import sleep
from tqdm import tqdm
from prettytable import PrettyTable
from prettytable.colortable import ColorTable, Themes
import plotly.express as px
from IPython.display import HTML # 导入HTML
import plotly.graph_objects as go
import plotly.io as pio
from package_xskj_NetworkXsimple import netGraph


pio.renderers.default = "iframe"
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
sns.set_theme()
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文




# 初始化netGraph神经网络绘图对象: 有向图
networkGraph = netGraph(type=1)


def plot_data_scatter(data_set):
    """
    desc：绘制原始数据的散点图
    paremeters:
        data_set pandas 数据集
    """
    # colors = ['red', 'blue']
    # fig = px.scatter(data_set, x=str(data_set.columns[0]), y=str(data_set.columns[-1]), color=str(data_set.columns[-1]))
    # 打印输出

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data_set.loc[data_set.iloc[:, -1] == data_set.iloc[:, -1].max()].iloc[:, 0],
        y=data_set.loc[data_set.iloc[:, -1] == data_set.iloc[:, -1].max()].iloc[:, -1],
        marker=dict(color="red", size=5),
        mode="markers",
        name="Positive",
    ))

    fig.add_trace(go.Scatter(
        x=data_set.loc[data_set.iloc[:, -1] == data_set.iloc[:, -1].min()].iloc[:, 0],
        y=data_set.loc[data_set.iloc[:, -1] == data_set.iloc[:, -1].min()].iloc[:, -1],
        marker=dict(color="blue", size=5),
        mode="markers",
        name="Negative",
    ))

    fig.update_layout(title="dataset scatter distribute",
                      xaxis_title=str(data_set.columns[0]),
                      yaxis_title=str(data_set.columns[-1]))

    return fig


def plot_feature_dim(data_set):
    """
    desc:不同维度的变量相关性分析
    paremeter:
        data_set  pandas 数据集
    """
    fig = px.scatter_matrix(
        data_set,
        height=1400,
        width=1400,
        title="dim relative for feature",
        dimensions=data_set.columns[1:],
        color=data_set.columns[-1]
    )
    fig.update_layout(font=dict(size=7))
    fig.update_xaxes(tickfont_family="Arial Black")
    fig.update_yaxes(tickfont_family="Arial Black")

    return fig


def plot_feature_y(X, X_label, Y):
    """
    desc: 特征量与真实值的相关性
    """
    m, n = X.shape

    axs = []
    # 设置画布
    fig = plt.figure(figsize=(14, 14), dpi=100)
    plt.subplots_adjust(bottom=0, right=0.8, top=1, hspace=0.5)
    # 列
    coloum = 3
    for i in range(n):
        ax = fig.add_subplot(math.ceil(n / coloum), coloum, i + 1)
        if i == 0:
            ax.set_ylabel('真实值 y')
        ax.set_xlabel('x')
        ax.set_title(X_label[i])

        # 绘制散点图
        ax.scatter(X[:, i], Y)
        # 绘制箱型图
        np.random.seed(10)  # 设置种子
        D = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))
        ax.boxplot(D, positions=[2, 4, 6], widths=1.5, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "white", "linewidth": 0.5},
                   boxprops={"facecolor": "C0", "edgecolor": "white",
                             "linewidth": 0.5},
                   whiskerprops={"color": "C0", "linewidth": 1.5},
                   capprops={"color": "C0", "linewidth": 1.5})

        axs.append(ax)


def plot_cost(cost):
    """
    desc:绘制损失值图
    """
    # fig, ax = plt.subplots()
    # ax.set_title("代价变化图")
    # ax.set_xlabel("iteration")
    # ax.set_ylabel("cost")
    # plt.plot(cost)
    # 最小损失
    # 求列表最大值及索引
    # min_value = min(cost) # 求列表最大值
    # min_idx = cost.index(min_value) # 求最大值对应索引

    # plot cost versus iteration
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))

    ax1.plot(np.arange(len(cost)), cost)
    index = math.ceil(len(cost) * 0.8)
    ax2.plot(index + np.arange(len(cost[index:])), cost[index:])

    ax1.set_title("Cost vs. iteration");
    ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost');
    ax2.set_ylabel('Cost')
    ax1.set_xlabel('iteration step');
    ax2.set_xlabel('iteration step')
    plt.show()


def plot_feature_weight(feature_labels, opt_W, opt_b):
    """
    desc：绘制模型特征权重信息
    paremeters:
        net       netGraph   netGraph对象
        feature_labels  list 特征名称
        opt_W     list  最优模型参数W
        opt_b     float 最优模型偏置b
    """
    # 转为数组list
    opt_W = opt_W.reshape(-1)


    for i in range(len(opt_W)):
        # 节点坐标(层，节点数)
        pos = (1, i + 1)
        # 节点名称
        name = f"x{i + 1}"
        # 节点标签
        label = feature_labels[i]
        # print(f"feature_labels={label}")
        # 增加网络节点
        networkGraph.addNode(
            name=name,
            pos=pos,
            label=label,
            label_color="red",
            nexts=[
                {
                    "node": "z(X)",
                    "label": f"w_{i+1}",
                    "color": "blue",  # 颜色
                    "weight": round(opt_b, 2)  # edge权重
                }
            ]
        )

    # 偏置节点
    networkGraph.addNode(
        name="b",
        pos=(1, len(opt_W) + 1),
        label="",
        label_color="red",
        nexts=[
            {
                "node": "z(X)",             #  连接节点
                "label": "b",               #  edge标签
                "color": "blue",           # 颜色
                "weight":  round(opt_b, 2)  # edge权重
            }
        ]
    )


def plot_cost_3d(W_opt, b_opt):
    """
    绘制3D损失代价图
    """

    def regression_model(X, W, b):
        # 线性组合 + sigmoid激活
        z = np.dot(X, W) + b
        return 1 / (1 + np.exp(-z))

    def loss_function(X, Y, W, b):
        # 交叉熵损失
        predictions = regression_model(X, W, b)
        loss = -np.mean(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
        return loss

    np.random.seed(0)

    # 根据 W_opt 的形状确定特征数量 m
    m = W_opt.shape[0]

    # 生成 m 维特征数据
    X = np.random.randn(1000, m)  # 生成 10000 行 m 列的随机特征矩阵

    # 随机生成标签数据 Y
    Y = np.random.randint(0, 2, size=1000)

    # 计算损失函数值
    Z = np.array([loss_function(X[i], Y[i], W_opt, b_opt) for i in range(X.shape[0])]).reshape(100, 100)

    # 绘制3D曲面图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面图
    X1 = np.linspace(-3, 3, 100)
    X2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(X1, X2)
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Loss')
    ax.set_title('3D Surface Plot of Loss Function')

    plt.show()


def plot_cost_contour(Cost):
    """
    绘制损失代价的等高线图
    """
    sns.set_theme(style="ticks")

    # Show the joint distribution using kernel density estimation
    data = pd.DataFrame({
        'W': pd.Series(np.linspace(-10, 10, len(Cost))),
        'b': pd.Series(np.linspace(-10, 10, len(Cost))),
        'cost': pd.Series(np.array(Cost))
    })
    g = sns.jointplot(
        data=np.array(Cost),
        x="W", y="b", hue="cost",
        kind="kde",
    )


def plot_missing(data_set):
    """
    desc:绘制缺失值分布
    paremeters:
        data_set  pandas 数据集
    """
    import missingno as msno
    msno.bar(data_set)


def plot_corr(data_set):
    """
    desc:绘制变量间的相关性关系
    """
    # 计算变量间的相关系数
    corr = data_set.corr()

    f, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("变量之间的相关性系数值")
    sns.heatmap(corr, annot=True, fmt=".2f", linewidths=.5, cmap="YlGn", ax=ax)

def Z_score_normalization(X):
    """
    desc:Z-score归一化
        公式：x* = ( x − μ ) / σ
    paremeters:
        X   np.array (m,n) 原始数据
    return:
        X_nor np.array (m,n)  归一化后的
    """
    # 计算样本的特征的均值和标准差
    Mu =    np.mean(X, axis=0)
    Sigma = np.std(X,  axis=0)

    # print(f"Mu = {Mu}")
    X_nor = (X - Mu) / Sigma

    return X_nor, Mu, Sigma


# 正则化后的回归模型
def data_dispose(data_set):
    """
    desc:  数据处理  利用pandas库进行处理 返回numpy对象
    parameters:
        data_set  pandas类型  数据集
    return
        X_dispose np.array （m, n）  处理后的特征量
        Y_dispose np.array  (m,1)    处理后的真实值
        data_labels   list    (,n)      特征标签

    """
    data_set = pd.DataFrame(data_set)

    # 提取特征量、特征标签、真实值
    X_dispose = data_set.iloc[:, :-1]  # 特征量 ：除最后一列外都为特征
    Y_dispose = data_set.iloc[:, -1]  # 真实值：最后一列为真实值
    data_labels = data_set.columns  # 特征标签

    return np.array(X_dispose), np.array(Y_dispose, ndmin=2).reshape(-1, 1, ), data_labels


def train_percentage():
    """
    desc:# 训练集占比
    """
    percent = 0.8

    return percent


def data_set_slice(data_set):
    """
    desc:数据集划分 8 2原则
            抽取样本原则:随机抽取样本比例
    paremeters:
       data_set pandas 数据集
    return:
       data_training_set pandas 训练集
       data_test_set     pandas 测试集

    """
    # 总样本数
    m = data_set.shape[0]

    # 训练集
    data_training_set = data_set.sample(frac=train_percentage())
    # 测试集
    data_test_set = data_set.drop(labels=data_training_set.index)

    return data_training_set, data_test_set


def init_W_b(X):
    """
    desc: 初始化w和b模型参数值
    parameters:
        X  np.array     （m, n）  特征数据
    return:
        W  np.array     （n,1）    模型参数值
        b  float                  模型参数指
    """
    n = np.array(X).shape[1]

    W = np.zeros((n, 1))
    b = 0.

    return W, b


def sigmod(z):
    """
    desc:激活函数

    """
    f = 1 / (1 + np.exp(-z))

    return f


def Hypothesis_function(X, W, b):
    """
    desc: 假设函数
    parameters:
        X  np.array     （m, n）  特征数据
        W  np.array     （n,1）    模型参数值
        b  float                  模型参数指
    returns:
        f_wb  np.array  (m,1)    预测值
    """

    # 线性模型
    h_wb = X @ W + b
    # 增加线性模型网络节点
    networkGraph.addNode(
        name="z(X)",
        pos=(2, 1),
        label="",
        label_color="red",
        nexts=[
            {
                "node": "g(z)",
                "label": "active fun",
            }
        ]
    )

    # sigmod函数激活
    f_wb = sigmod(h_wb)

    # 增加sigmod函数激活网络节点
    networkGraph.addNode(
        name="g(z)",
        pos=(3, 1),
        label=r"sigmod(z) = \frac{1}{ 1 + e^{-z}}",
        label_color="green",
        nexts=[
            {
                "node": "f(g)",
                "label": "model output",
            }
        ]
    )

    # 输出网络节点
    networkGraph.addNode(
        name="f(g)",
        pos=(4, 1),
        label="P(y|x;W,b)",
        label_color="green"
    )


    return f_wb


def regularize_lambda():
    """
    desc:给出正则系数
    lambda$大  ，则W_j小,惩罚大
    lambda$小  ，则W_j大
    return:
        _lambda  float  正则系数
    """
    _lambda = 0.5

    return _lambda


def cost_function(X, Y, W, b):
    """
    desc：代价函数
    parameters:
        X  np.array （m, n）    特征数据
        Y  np.array  (m,1)     真实值
        W  np.array  (n,1)     模型参数值
        b  float                模型参数值
    return:
        J_w_b  float            成本/代价
        Err    np.array  (m,1)  损失
    """
    m = np.array(X).shape[0]

    # 代价cost
    f_wb = Hypothesis_function(X, W, b)
    Err = f_wb - Y
    Loss = -Y * np.log(f_wb) - (1 - Y) * np.log(1 - f_wb)
    cost = (1 / m) * np.sum(Loss)

    # 正则regularize
    _lambda = regularize_lambda()  # 正则系数
    regularize = (_lambda / (2 * m)) * np.sum(W ** 2)
    # 成本 = cost + regularize
    J_wb = cost + regularize

    return J_wb, Err


def compute_gradient_descent(X, Y, W, Err):
    """
    desc:计算正则化后的梯度(偏导)
    parameters:
        X  np.array （m, n）    特征数据
        Y  np.array  (m,1)      真实值
        W  np.array  (n,1)      模型参数值
        Err    np.array  (m,1)  损失
    return:
        dJ_dW np.array  (n,1)  J对w的偏导数
        dJ_db float            J对b的偏导数
    """
    m = np.array(X).shape[0]
    _lambda = regularize_lambda()

    # 计算偏导数
    tmp_dJ_dW = (1 / m) * np.dot(X.T, Err) + (_lambda / m) * W
    tmp_dJ_db = (1 / m) * np.sum(Err)

    # 同时更新
    dJ_dW = tmp_dJ_dW
    dJ_db = tmp_dJ_db

    return dJ_dW, dJ_db


def fit(X_train, Y_train, lr=0.01, iteration=10000, data_label=[]):
    """
    desc:模型训练，模型拟合
    parameters:
        X_train  np.array （m, n）    训练集的特征数据
        Y_train  np.array  (m,1)      训练集的真实值
        lr  float  学习率 默认0.1
        iteration int 迭代次数 默认10000
        data_label  数据标签 （包含y真实标签，最后一列） list
    return:
        W_opt = W
        b_opt = b
    """
    # 数据处理
    X, Y = X_train, Y_train

    # 损失
    Cost = []
    # 存储模型参数
    W_array = []
    b_array = []

    # 初始化模型参数
    W, b = init_W_b(X)

    for index in tqdm(range(iteration), desc="model training", total=iteration, unit="iteration",
                      postfix={'regularized paremeter': regularize_lambda()}):
        # 1.计算cost，losss
        J_wb, Err = cost_function(X, Y, W, b)

        # 放进数组中存储
        W_array.append(W)
        b_array.append(b)
        Cost.append(J_wb)
        ##############输出打印##############
        # print(f"iteration {index}: cost = {J_wb}")

        # 2.计算梯度
        gradient_W, gradient_b = compute_gradient_descent(X, Y, W, Err)

        # 3.模型参数更新
        W -= lr * gradient_W
        b -= lr * gradient_b

    # 最小损失
    # 求列表最大值及索引
    min_value = min(Cost)  # 求列表最大值
    min_idx = Cost.index(min_value)  # 求最大值对应索引

    # 最优点
    W_opt = W_array[min_idx]
    b_opt = b_array[min_idx]

    # 绘制cost
    # plot_cost(Cost)
    plot_cost(Cost)
    # 绘制特征权重图
    plot_feature_weight(data_label, W_opt, b_opt)

    return W_opt, b_opt, (min_idx, min_value)


def predict(X, W, b):
    """
    desc:模型预测
    parameters:
        X  np.array     （m, n）  特征数据
        W  np.array     （n,1）    模型参数值
        b  float                  模型参数指
    """
    predict_y = Hypothesis_function(X, W, b)

    return predict_y


def evaluate(data_test, W, b, mu=0, sigma=0):
    """
    desc:模型评价
    """
    # 数据处理
    X_test, Y_test, X_labels = data_dispose(data_test)

    # 决策阈值
    threshold = 0.5

    # 预测
    predict = Hypothesis_function(X_test, W, b).reshape(-1).tolist()
    predict_result = []
    for i in range(len(predict)):
        if predict[i] > threshold:
            predict_result.append(1)
        else:
            predict_result.append(0)

    # 比较
    result = np.abs(np.array(predict_result) - Y_test.T)

    # 统计正确个数
    err_count = np.sum(result)
    correct_count = len(predict) - err_count

    # print(f"correct count: {correct_count}   error count: {err_count}")
    # 正确率
    correct_rate = correct_count / len(predict)

    return correct_rate, predict_result, predict, Y_test.T.tolist(), correct_count, err_count



if __name__ == '__main__':
    # 加载数据集
    column_names = [
        'Sample code number', 'Clump Thickness', ' Cell Size',
        'Cell Shape', 'Marginal Adhesion',
        'Cell Size', 'Bare Nuclei', 'Bland Chromatin',
        'Normal Nucleoli', 'Mitoses', 'Class'
    ]
    data_set = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
        names=column_names)
    # 绘制缺失值分布
    plot_missing(data_set)

    # object转float
    data_set["Bare Nuclei"] = pd.to_numeric(data_set["Bare Nuclei"], errors='coerce')

    # 数据预处理:注意这里数据处理只能在上面一步把数据类型都转化为数字型的才能进行缺失值判断，因为判断函数仅认数字型!!!!!!!!!!!容易出错
    # 1.缺失值处理：替换 or 删除
    data_set.fillna(0, inplace=True)
    # 丢弃带有缺失值的数据（只要有一个维度有缺失）
    # data_set.dropna(how='any', inplace=True)

    # 将标签替换成0 或 1
    min_value = data_set.iloc[:, -1].min()
    max_value = data_set.iloc[:, -1].max()
    data_set.iloc[:, -1].replace([min_value, max_value], [0, 1], inplace=True)

    # 数据集划分
    data_training_set, data_test_set = data_set_slice(data_set)

    # 绘制数据散点图分布
    data_scatter_fig = plot_data_scatter(data_training_set)
    HTML(data_scatter_fig.to_html())

    # 绘制相关性变量
    feature_fig = plot_feature_dim(data_training_set)
    HTML(feature_fig.to_html())

    # 删掉编号列
    data_training_set.drop('Sample code number', axis=1, inplace=True)
    data_test_set.drop('Sample code number', axis=1, inplace=True)  # 训练集
    X, Y, data_labels = data_dispose(data_training_set)

    # 绘制特征分量关于真实值的散点图
    plot_feature_y(X, data_labels, Y)
    # 绘制变量相关性热力图
    plot_corr(data_set)

    # 规范化
    # X , mu , sigma =Z_score_normalization(X)

    # 数据训练拟合
    lr = 0.01  # learning rate
    iteration = 1000  # iteration
    W_opt, b_opt, (min_idx, min_value) = fit(X, Y, lr, iteration, data_labels)

    # 绘制模型的网络图
    networkGraph.draw()

    print("traing result:")
    table_fit = ColorTable(theme=Themes.OCEAN, field_names=['lr', 'iteration', 'regularized paremeter', 'feature count',
                                                            'train_data percent', 'min cost', 'opt_W', 'opt_b'])
    table_fit.add_row([lr, iteration, regularize_lambda(), X.shape[1], train_percentage(),
                       f'index={min_idx} cost={round(min_value, 5)}', W_opt, b_opt])
    print(table_fit)

    # 模型评价：测试集
    correct_rate, predict_result, predict, y, correct_count, err_count = evaluate(data_test_set, W_opt, b_opt, mu=0,
                                                                                  sigma=0)
    print("model evaluate:")
    table_evaluate = PrettyTable(['accuracy', 'correct count', 'error count'])
    table_evaluate.add_row([round(correct_rate, 4), correct_count, err_count])
    print(table_evaluate)


