import numpy as np
import pandas as pd
from neuralnetwork import NeuralNetwork
from numpy import genfromtxt
from cross_validation import train_test_split, accuracy_score
from sklearn.metrics import roc_curve, auc
import timeit
import matplotlib.pyplot as plt
import pandas as pandas

eeg_data = pd.read_csv('EEG_Eye_State.csv', header=None)

# remove examples with extreme outliers (only 3 rows out of 15,000)
eeg_data = eeg_data[(eeg_data <= 10000).all(axis=1)]

X = eeg_data.iloc[:,:-1]
y = eeg_data.iloc[:,-1].reshape(-1,1)
print X.shape
print y.shape

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

# use my custom data splitter for training/test
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=0, test_size = 0.25)


start_time = timeit.timeit()
print "starting fit"


nn = NeuralNetwork(n_iter = 2, n_print = 5, learning_rate = 1,\
                 num_hidden_units=24, seed=42, verbose = False)

nn.fit(X_train, y_train)
end_time = timeit.timeit()
print "Fitting time: {}".format(end_time - start_time)
print "W matrix (size = {} x {}) = {}".format(nn.W.shape[0],nn.W.shape[1],nn.W)
print "V matrix (size = {} x {}) = {}".format(nn.V.shape[0],nn.V.shape[1],nn.V)


np.set_printoptions(threshold=np.inf)
y_pred = nn.predict(X_train)

print "Training Accuracy score = {}".format(accuracy_score(y_train, y_pred))

y_pred = nn.predict(X_test)
print "Test Accuracy score = {}".format(accuracy_score(y_test, y_pred))

y_prob = nn.predict_prob(X_test, add_bias_unit=True)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print "roc_auc = {}".format(roc_auc)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for EEG Neural Network')
plt.legend(loc="lower right")
plt.savefig('EEG_ROC.png')
 
