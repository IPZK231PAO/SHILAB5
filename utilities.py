# utilities.py
import matplotlib.pyplot as plt
import numpy as np

def visualize_classifier(classifier, X, y, title):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.show()
