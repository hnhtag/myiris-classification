import numpy as np
from collections import Counter

def educlidean_distance(p,q):
    return np.sqrt(np.sum((p-q)**2))

class MyKNN:
    def __init__(self,k=5):
        self.k = k
    def fit(self,x,y):
        self.X = x
        self.y = y
    def predict(self,test):
        predictions = [self.calculate(point) for point in test]
        return predictions
    def calculate(self,point):
        distance = [educlidean_distance(each_x,point) for each_x in self.X]
        k_label_group = np.argsort(distance)[:self.k]
        k_label = [self.y[i] for i in k_label_group]
        most_common = Counter(k_label).most_common(1)
        return most_common[0][0]