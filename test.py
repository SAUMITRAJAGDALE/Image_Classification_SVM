import matplotlib.pyplot as plt 
import numpy as np
from sklearn import datasets
from sklearn import svm

digits= datasets.load_digits()
clf =svm.SVC(gamma=0.0001,C=100)
x,y= digits.data[:-1], digits.target[:-1]
print(x)
print(len(digits))



