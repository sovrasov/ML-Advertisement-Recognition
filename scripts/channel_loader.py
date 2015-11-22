from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn import feature_selection
mem = Memory("../Dataset")

@mem.cache
def get_small_data(channel):
    data = load_svmlight_file("../Dataset/" + channel + ".txt")
    selector = feature_selection.VarianceThreshold()
    X = selector.fit_transform(data[0])
    return X, data[1]

def get_data(channel):
    data = load_svmlight_file("../Dataset/" + channel + ".txt")
    return data[0], data[1]