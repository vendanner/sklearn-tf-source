from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    """
    集成学习方法和单纯模型性能比较
    """
    X,y = make_moons(n_samples=1000,noise=0.3,random_state=42)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

    svc_clf = SVC(random_state=42)
    lr_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(random_state=42)
    # 硬投票
    voting_clf = VotingClassifier(estimators=[('lr',lr_clf),('svc',svc_clf),('rnd',rnd_clf)],voting='hard')

    for clf in (svc_clf,lr_clf,rnd_clf,voting_clf):
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        # 分类正确概率
        score = accuracy_score(y_test,y_pred)
        print(clf.__class__.__name__,score)

    """
    RandomForestClassifier 0.952
    VotingClassifier 0.936
    SVC 0.94
    LogisticRegression 0.868
    总体效果来说，LR最差，RandomForestClassifier最好，但请注意随机森林也是集成方法
    """