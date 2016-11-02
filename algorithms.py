from sklearn.pipeline import Pipeline
#six different algorithms all using piplines

def gaussNB(scaler,skb):

    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB()
    clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('NaiveBayes', gnb)])

    return clf

def SVC(scaler,skb):

    from sklearn.svm import SVC

    svc = SVC(C=100)
    clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('SVC', svc)])

    return clf

def DTree(scaler,skb):

    from sklearn import tree
    dt = tree.DecisionTreeClassifier()
    clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('Decision Tree', dt)])

    return clf


def LogReg(scaler,skb):

    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression()
    clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('Logistic Regression', lg)])

    return clf


def LinearS(scaler,skb):

    from sklearn.svm import LinearSVC

    lsvc = LinearSVC()
    clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('Linear SVC', lsvc)])


    return clf

def RandForest(scaler,skb):

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('Random Forest', rf)])

    return clf
