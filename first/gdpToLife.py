#!/usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import preprocessing
from sklearn import pipeline

def save_fig(fig_id, tight_layout=True):
    """
    保存当前plt 绘制的图片
    :param fig_id: 
    :param tight_layout: 
    :return: 
    """
    path = "images"+fig_id + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def prepare_country_stats(oecd_bli,gdp_per_captia):
    """
    oecd_bli 和 gdp_per_captia 数据利用"Country"整合成一个数据
    :param oecd_bli: 
    :param gdp_per_captia: 
    :return: 
    """
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    # 得到 行="Country"，列="Indicator" oecd_bli
    oecd_bli = oecd_bli.pivot(index = "Country",columns = "Indicator",values = "Value")

    gdp_per_captia.rename(columns = {"2015":"GDP per capita"},inplace=True)
    gdp_per_captia.set_index("Country",inplace = True)

    full_country_stats = pd.merge(left = oecd_bli,right = gdp_per_captia,left_index = True,right_index = True)
    full_country_stats.sort_values(by = "GDP per capita",inplace = True)

    remove_indices = [0,1,6,8,33,34,35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita","Life satisfaction"]],full_country_stats[["GDP per capita","Life satisfaction"]].iloc[keep_indices],full_country_stats[["GDP per capita","Life satisfaction"]].iloc[remove_indices]

def learnModel(country_stats):
    """
    训练模型
    :param country_stats: 测试数据
    :return: 模型
    """
    X = np.c_[country_stats["GDP per capita"]]
    y = np.c_[country_stats["Life satisfaction"]]
    # 选择模型
    model = sklearn.linear_model.LinearRegression()

    # 训练数据
    model.fit(X,y)
    return model

def predictAndShowGdpLife(country_stats):
    """
    绘制以gdp 为X轴，life 为Y轴的点；并在最后预测了gdp = 22587 的life statisfaction
    :param country_stats: 
    :return: 
    """

    # 可视化数据
    country_stats.plot(kind = "scatter",x = "GDP per capita",y = "Life satisfaction")
    # 保存数据
    country_stats.to_csv("lifesat.csv")

    # 测试模型
    X_new = [[22587]]
    print(learnModel(country_stats).predict(X_new))

def showFullDataModel(full_data,country_stats,miss_data):
    """
    展示不同数据量生成的模型，模型调整来适应所有数据
    看生成结果imagesrepresentative_training_data_scatterplot.png，着重看miss_data(红色标注)
    :param full_data: 全量数据
    :param country_stats: 有删减数据
    :param miss_data: 被删减数据
    :return: 
    """
    # 额外数据
    position_text2 = {
        "Brazil": (1000, 9.0),
        "Mexico": (11000, 9.0),
        "Chile": (25000, 9.0),
        "Czech Republic": (35000, 9.0),
        "Norway": (60000, 3),
        "Switzerland": (72000, 3.0),
        "Luxembourg": (90000, 3.0),
    }
    country_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction", figsize=(8,3))
    # 设定展示坐标轴的范围
    plt.axis([0,110000,0,10])

    # miss_data 数据展示准备
    for country,pos_text in position_text2.items():
        pos_data_x ,pos_data_y = miss_data.loc[country]
        plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
                     arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
        plt.plot(pos_data_x, pos_data_y, "rs")

    # country_stats 数据训练出来的线性模型，虚线
    X = np.linspace(0,110000,100)
    #  w = (w_1,..., w_p) 作为 coef_ ，定义 w_0 作为 intercept_
    # 本案例中特征向量只有1维
    model = learnModel(country_stats)
    w0,w1 = model.intercept_[0],model.coef_[0][0]
    plt.plot(X,w1*X+w0,"b:")

    # full_data 数据训练出来的线性模型，实线
    model_full = learnModel(full_data)
    w0full,w1full = model_full.intercept_[0],model_full.coef_[0][0]
    plt.plot(X, w1full * X + w0full, "k")

    save_fig('representative_training_data_scatterplot')

def showModelOverfitting(full_data):
    """
    模型过拟合情况,详情看
    :param full_data: 训练数据overfitting_model_plot.png
    :return: 
    """
    full_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(8, 3))
    plt.axis([0, 110000, 0, 10])

    # 暂时不清楚，后续回来看
    poly = preprocessing.PolynomialFeatures(degree = 60,include_bias=False)
    scaler = preprocessing.StandardScaler()
    module = sklearn.linear_model.LinearRegression()

    # 数据预处理太过导致过拟合
    Xfull = np.c_[full_data["GDP per capita"]]
    yfull = np.c_[full_data["Life satisfaction"]]
    X = np.linspace(0, 110000, 1000)

    pipeline_reg = pipeline.Pipeline([('poly', poly), ('scal', scaler), ('lin', module)])
    pipeline_reg.fit(Xfull, yfull)
    curve = pipeline_reg.predict(X[:, np.newaxis])
    plt.plot(X, curve)
    save_fig('overfitting_model_plot')

def showRidgeRinLinearModel(full_data,country_stats,miss_data):
    """
    比对不同模型之间的性能 - 拟合能力
    在训练数据都是country_stats 情况下，ridge 优化的线性模型比没优化之前的模型对于miss_data有更强的拟合
    详细看ridge_model_plot.png
    :param full_data: 全量训练数据
    :param country_stats: 有删减训练数据
    :param miss_data: 被删减的训练
    :return: 
    """
    plt.figure(figsize=(8, 3))
    plt.xlabel("GDP per capita")
    plt.ylabel('Life satisfaction')
    plt.plot(list(country_stats["GDP per capita"]), list(country_stats["Life satisfaction"]), "bo")
    plt.plot(list(miss_data["GDP per capita"]), list(miss_data["Life satisfaction"]), "rs")

    X = np.linspace(0,110000,1000)
    full_model = learnModel(full_data)
    country_model = learnModel(country_stats)

    # ridge
    Xcountry = np.c_[country_stats["GDP per capita"]]
    ycountry = np.c_[country_stats["Life satisfaction"]]
    # alpha = 10的9.5次方;alpha就是超参 -- alpha*（W1平方 + W2平方 ... Wn平方）
    ridge_mode = sklearn.linear_model.Ridge(alpha=10**9.5)
    ridge_mode.fit(Xcountry,ycountry)

    # 画出三个模型的线
    plt.plot(X, full_model.coef_[0][0] * X + full_model.intercept_, "r--", label="Linear model on all data")
    plt.plot(X, country_model.coef_[0][0] * X + country_model.intercept_, "b:", label="Linear model on partial data")
    plt.plot(X, ridge_mode.coef_[0][0] * X + ridge_mode.intercept_, "b", label="Regularized linear model on partial data")

    plt.legend(loc="lower right")
    plt.axis([0, 110000, 0, 10])
    save_fig('ridge_model_plot')



if __name__ == "__main__":

    # 设置pd 输出全显示
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 100)
    pd.set_option('display.width', 1000)

    #  加载数据
    oecd_bli = pd.read_csv("oecd_bli_2015.csv",thousands=',')
    gdp_per_captia = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',encoding='latin1',na_values="n/a")
    # 准备数据
    full_data,country_stats,miss_data = prepare_country_stats(oecd_bli,gdp_per_captia)

    # 预测和展示(gdp life) 点
    # predictAndShowGdpLife(country_stats)
    # 通过此案例可以发现训练数据量对于模型的重要性
    # showFullDataModel(full_data,country_stats,miss_data)
    # 展示模型过拟合的情况
    # showModelOverfitting(full_data)
    # 展示带Ridge 优化的线性回归
    showRidgeRinLinearModel(full_data,country_stats,miss_data)

    # 显示
    plt.show()