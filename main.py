#%%

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


x, y = make_moons(n_samples=2500, noise=.0)

def grid(X, h=.02):
    x_min, x_max = X[:,0].min() - 1, X[:,1].max() + 2
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_boundary(model, X, y):
    xx, yy = grid(X)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=.8, cmap=plt.cm.rainbow)
    plt.scatter(X[:,0], X[:,1], c=y, s=40, edgecolors='k', cmap=plt.cm.RdBu)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2)

linear = LogisticRegression().fit(X_train, y_train)
nonlinear = SVC(kernel='rbf', gamma=1).fit(X_train, y_train)

#%%
plt.figure(figsize=(12,5))
plt.title('Linearity boundaries')
plt.subplot(1,2,1)
plot_boundary(linear, X_train, y_train)
plt.subplot(1,2,2)
plot_boundary(nonlinear, X_train, y_train)
plt.tight_layout()
plt.show()