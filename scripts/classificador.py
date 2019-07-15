import os
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels import robust
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from plot_confusion import plot_confusion_matrix

# open files and get a 100 sample from each
dirpath = os.getcwd()
df = pd.read_csv(os.path.join(dirpath + "/data/GastroTCGA.tsv"), delimiter='\t', index_col=0, header=None)

# extract features
X = df.T.iloc[:, 2:].values

# select classes
Y = df.T.iloc[:, 1].values

##### Simple Validation #####
# 30% of the data in the test set and 70% in the training set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# call logistic regression
logreg = LogisticRegression(penalty='l2', C=0.2, solver='lbfgs', multi_class='multinomial', max_iter = 2000)

# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X_train, Y_train)

# Apply the model on the test set
Y_pred = logreg.predict(X_test)
probs = logreg.predict_proba(X_test)

# Accuracy
print('Misclassified test samples: %d' % (Y_test != Y_pred).sum())
print('Accuracy: %.2f' % metrics.accuracy_score(Y_test, Y_pred))

# Make some plots
class_names = ['COAD', 'READ', 'STAD']

# Plot non-normalized confusion matrix
plot = plot_confusion_matrix(Y_test, Y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

plot.savefig(os.path.join(dirpath + "/figures/confusion.png"))

# Plot normalized confusion matrix
plot_normalized = plot_confusion_matrix(Y_test, Y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plot_normalized.savefig(os.path.join(dirpath + "/figures/confusion_normalized.png"))

# Make some heatmaps
dfProbs = pd.DataFrame(probs)

dfProbs.columns = class_names
dfProbs.index = Y_test

plt.figure(figsize=(10, 200))
sns.clustermap(dfProbs.T, vmax=1, cmap="YlGnBu");
plt.savefig(os.path.join(dirpath + "/figures/heatmap.png"))

##### Cross Validation #####
# evaluate the model using 10-fold cross-validation
scores = cross_val_score(logreg, X, Y, scoring='accuracy', cv=5)
Y_pred = cross_val_predict(logreg, X, Y, cv=5)
probs = cross_val_predict(logreg, X, Y, cv=5, method='predict_proba')

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Plot non-normalized confusion matrix
plot_cross = plot_confusion_matrix(Y, Y_pred, classes=class_names,
                      title='Confusion matrix, Cross Validation, without normalization')

plot_cross.savefig(os.path.join(dirpath + "/figures/cross_confusion.png"))

# Plot normalized confusion matrix
plot_cross_normalized = plot_confusion_matrix(Y, Y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix, Cross Validation')

plot_cross_normalized.savefig(os.path.join(dirpath + "/figures/cross_confusion_normalized.png"))

# Make some heatmaps
dfProbs = pd.DataFrame(probs)

dfProbs.columns = class_names
dfProbs.index = Y

plt.figure(figsize=(10, 200))
sns.clustermap(dfProbs.T, vmax=1, cmap="YlGnBu");
plt.savefig(os.path.join(dirpath + "/figures/cross_heatmap.png"))

plt.figure(figsize=(10, 200))
sns.clustermap(dfProbs.T, vmax=1, cmap="YlGnBu", col_cluster=False);
plt.savefig(os.path.join(dirpath + "/figures/cross_heatmap_colcluster.png"))
