
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data = pd.read_csv("./data/info0202.csv",header=None)
data.columns = ["days_exposure","days_positive","days_symp","test_result_YDA","test_YDA","symp_num","positive_num","clustersize","action"]
state = data.drop(columns = ["action"])
# state = data.drop(columns = ["symp_num","clustersize","positive_num","action"])
action = data["action"]
X_train, X_test, y_train, y_test = train_test_split(state, action, test_size=0.30)


# for i in range(3,10):
#     clf = DecisionTreeClassifier(criterion='gini', max_depth=i, splitter="best",random_state=2000)
#     clf = clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     feature_names = X_train.columns
#     feature_importance = pd.DataFrame(clf.feature_importances_, index=feature_names).sort_values(0, ascending=False)
#     print(feature_importance)

#     from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
#     y_prob0 = clf.predict_proba(X_train)
#     train_auc = roc_auc_score(y_train, y_prob0, multi_class='ovr')
#     y_prob1 = clf.predict_proba(X_test)
#     test_auc = roc_auc_score(y_test, y_prob1, multi_class='ovr')
#     acc = accuracy_score(y_test,y_pred)
#     # print(".............")
#     print(train_auc)
#     print(test_auc)
#     print(acc)
#     print(f1_score(y_test, y_pred, average='weighted'))


clf = DecisionTreeClassifier(criterion='gini', max_depth=10, splitter="best")
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
feature_names = X_train.columns
feature_importance = pd.DataFrame(clf.feature_importances_, index=feature_names).sort_values(0, ascending=False)
print(feature_importance)

from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
y_prob0 = clf.predict_proba(X_train)
train_auc = roc_auc_score(y_train, y_prob0, multi_class='ovr')
y_prob1 = clf.predict_proba(X_test)
test_auc = roc_auc_score(y_test, y_prob1, multi_class='ovr')
acc = accuracy_score(y_test,y_pred)
    # print(".............")
print(train_auc)
print(test_auc)
print(acc)
print(f1_score(y_test, y_pred, average='weighted'))
# # print( classification_report(y_test, y_pred) )
joblib.dump(clf, "./decision_tree/DT_0202.pkl") 
from sklearn import tree
fig = plt.figure(figsize=(100,30))
_ = tree.plot_tree(clf, feature_names=feature_names, class_names={0:'No Quarantine&No Test', 1:'Quarantine&No Test', 2:'No Quarantine&Test', 3: 'Quarantine&Test', 4:'Test + quarantine, test - no quarantine'}, filled=True, rounded=True, fontsize=23)
# plt.show()
plt.savefig('./decision_tree/DT_0202.png')