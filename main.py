#%% Prelim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split

# TODO: 3D mapping for make circles

x, y = make_moons(n_samples=2500, noise=.05)
x_circles, y_circles = make_circles(n_samples=5000, factor=0.5, noise=0.05)

def grid(X, h=.02):
    """_summary_

    Args:
        X (_type_): Datapoints in space
    """
    x_min, x_max = X[:,0].min() - 1, X[:,1].max() + 1.2
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_boundary(model, X, y):
    """_summary_

    Args:
        model (_type_): Linear or nonlinear model used
        X (_type_): Datapoints in space
        y (_type_): Set of moon point colors
    """
    xx, yy = grid(X)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=.8, cmap=plt.cm.rainbow)
    plt.scatter(X[:,0], X[:,1], c=y, s=40, edgecolors='k', cmap=plt.cm.rainbow)


linear = LogisticRegression().fit(x, y)
nonlinear = SVC(kernel='rbf', gamma=1).fit(x, y)


#%% Plot moon boundaries
plt.figure(figsize=(12,6), dpi=400)
plt.title('Linearity boundaries')

plt.subplot(1,2,1)
plot_boundary(linear, x, y)

plt.subplot(1,2,2)
plot_boundary(nonlinear, x, y)

plt.tight_layout()
plt.show()

#%% Plot circle boundaries
linear = LogisticRegression().fit(x_circles, y_circles)
nonlinear = SVC(kernel='rbf', gamma=1).fit(x_circles, y_circles)

plt.figure(figsize=(12,6), dpi=400)
plt.title('Linearity boundaries')

plt.subplot(1,2,1)
plot_boundary(linear, x_circles, y_circles)

plt.subplot(1,2,2)
plot_boundary(nonlinear, x_circles, y_circles)

plt.tight_layout()
plt.show()

