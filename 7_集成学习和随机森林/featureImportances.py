from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt

def plt_digit(data):
    """
    
    :param data: 
    :return: 
    """
    # 一张图片是28*28=768 维
    image = data.reshape(28,28)
    plt.imshow(image,cmap=matplotlib.cm.hot,interpolation="nearest")
    plt.axis("off")

if __name__ =="__main__":
    """
    训练随机树后输出特征重要度
    """
    mnist = fetch_mldata("MNIST original", data_home="dataSet/")
    rnd_clf = RandomForestClassifier(random_state=42)
    rnd_clf.fit(mnist["data"],mnist["target"])

    plt_digit(rnd_clf.feature_importances_)
    cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
    cbar.ax.set_yticklabels(['Not important', 'Very important'])
    plt.show()
    """
    观察图片，最中心的位置颜色最亮，特征最重要，表示数字集中显示在最中间
    """