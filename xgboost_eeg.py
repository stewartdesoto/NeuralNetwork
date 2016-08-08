import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

data = np.loadtxt('EEG_Eye_State.csv', delimiter=',')
# print data[0]
X = data[:,0:-1]
y = data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
dtrain = xgb.DMatrix( X_train, label=y_train)
dtest = xgb.DMatrix( X_test, label=y_test)
# print dtrain

gbm = xgb.XGBClassifier(max_depth=10, n_estimators=1000, learning_rate=0.1)
gbm.fit(X_train, y_train)
predictions_train = gbm.predict(X_train)

# print y[0:100]
# print predictions[0:100]
print accuracy_score(y_train, predictions_train)

predictions_test = gbm.predict(X_test)
print accuracy_score(y_test, predictions_test)

params = [{'max_depth': [8, 10, 12], 'n_estimators': [800, 1000, 1600], 'learning_rate': [0.07, 0.1, 0.2, 0.3]}]
#params = [{'learning_rate': [0.1, 0.3]}]
clf = GridSearchCV(xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.5), cv=10, param_grid = params, verbose=4)
clf.fit(X_train, y_train)

print clf.grid_scores_
print clf.best_estimator_
print clf.best_score_
print clf.best_params_
print clf.scorer_

predictions_test = clf.predict(X_test)
print accuracy_score(y_test, predictions_test)

'''
0.942234089898
{'n_estimators': 1000, 'learning_rate': 0.3, 'max_depth': 10}
<function _passthrough_scorer at 0x112278500>
0.950867823765
'''