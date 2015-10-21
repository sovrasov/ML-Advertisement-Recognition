from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn import neighbors
mem = Memory("../Dataset")

@mem.cache
def get_data(channel):
    data = load_svmlight_file("../Dataset/"+channel+".txt")
    return data[0], data[1]

n_neighbors = 5

XB, yB = get_data('BBC')
XC, yC = get_data('CNN')

clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(XB, yB)

print 'Number of BBC frames = ' + str(XB.shape[0]) + '\n'
print 'Number of CNN frames = ' + str(XC.shape[0])

score = clf.score(XC[:10000], yC[:10000])
print 'score = ' + str(score)