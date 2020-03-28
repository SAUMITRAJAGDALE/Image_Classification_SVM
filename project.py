import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn import svm
i=int(input("ENTER ANY RANDOM NUMBER LESS THAN 1700: "))
digits= datasets.load_digits()
clf =svm.SVC(gamma=0.0001,C=100)
x,y= digits.data[:-1], digits.target[:-1]
clf.fit(x,y)
print("PREDICTION: ", clf.predict(digits.data[[i]]))
plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()



