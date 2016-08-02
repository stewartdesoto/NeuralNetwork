import numpy as np

def train_test_split(X, y, seed=0, test_size = 0.2):
  if len(X) != len(y):
    raise Exception("Mismatch between length of input and label arrays.")
  ind = np.random.permutation(len(y))
  X , y = X[ind], y[ind]
  n = len(y)
  num_train = (1 - test_size) * n
  return X[:num_train], X[num_train:], y[:num_train], y[num_train:]


def accuracy_score(y_true, y_pred):
  if len(y_true) != len(y_pred):
    raise Exception("Arrays don't have same length")
  num_correct = np.sum(y_true == y_pred)
  print "num_correct = {}".format(num_correct)
  print "total = {}".format(len(y_true) * y_true.shape[1])
  # in case of multiple outputs, divide total correct by number of outputs
  return float(num_correct) / (len(y_true) * y_true.shape[1])
