import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def display_lot(X,y):
    """
    
    :param X: 
    :param y: 
    :return: 
    """
    # 展示散点
    plt.plot(X,y,'.')
    plt.axis([-3, 3, 0, 10])

    # 训练模型拟合散点
    # 最高2次方
    ploy_features = PolynomialFeatures(degree=2,include_bias=False)
    # 多加了个特征X平方，现在是2个特征值X，X平方,可以拟合非线性数据
    X_ploy = ploy_features.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_ploy,y)
    # 画出LinearRegression 拟合后的线,变为2维数组
    x = np.linspace(-3,3,100).reshape(100,1)

    plt.plot(x,lin_reg.predict(ploy_features.fit_transform(x)),color = 'r',label='2')

    # 单纯的线性模型
    lin_cfg = LinearRegression()
    lin_cfg.fit(X,y)
    plt.plot(x, lin_cfg.predict(x), color='b', label='1')

    # 300次方的多项式
    ploy_features300 = PolynomialFeatures(degree=300, include_bias=False)
    X_ploy300 = ploy_features300.fit_transform(X)
    lin_reg300 = LinearRegression()
    lin_reg300.fit(X_ploy300,y)
    plt.plot(x, lin_reg300.predict(ploy_features300.fit_transform(x)), color='g', label='300')
    # 单纯线性模型欠拟合，300次方过拟合，2次方差不多；交叉验证和网格搜索，去调试出最佳参数
    plt.legend()
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title("polynomial")
    plt.show()

def plot_learning_curves(model,X,y):
    """
    计算误差
    :param model: 
    :param X: 
    :param y: 
    :return: 
    """
    X_train,X_val,y_trian,y_val = train_test_split(X,y,test_size=0.2)
    train_errors,val_errors = [],[]
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_trian[:m])
        y_trian_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_trian[:m],y_trian_predict))
        val_errors.append(mean_squared_error(y_val,y_val_predict))
    plt.plot(np.sqrt(train_errors),'r-+',linewidth=2,label = "train")
    plt.plot(np.sqrt(val_errors),'b-',linewidth=1,label = "val")
    plt.xlabel("Train set size")
    plt.ylabel("RMSE")

def analysis_polynomial(X,y):
    """
    学习率分析多项式回归
    :param X: 
    :param y: 
    :return: 
    """
    polynomial_pipeline = Pipeline([
        ("poly_feature",PolynomialFeatures(degree=10,include_bias=False)),
        ("linear",LinearRegression())
    ])
    lin_cfg = LinearRegression()
    plot_learning_curves(polynomial_pipeline,X,y)
    plt.axis([0,80,0,3])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    """
    多项式回归
    """
    m = 100;
    X = 6 * np.random.rand(m,1) - 3
    # 含X平方变非线性，并加入随机噪声
    y = 0.5 * X**2 + X + 2 + np.random.rand(100,1)

    # display_lot(X, y)
    analysis_polynomial(X,y)