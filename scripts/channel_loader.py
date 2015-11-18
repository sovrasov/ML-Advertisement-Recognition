from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
mem = Memory("../Dataset")

@mem.cache
def get_data(channel):
    data = load_svmlight_file("../Dataset/" + channel + ".txt")
    return data[0], data[1]
