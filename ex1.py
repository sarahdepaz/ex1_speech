import numpy as np
import os
import librosa
from sklearn.neighbors import KNeighborsClassifier

def dtw(x, y, dist):
    assert len(x)  # Report error while x is none
    assert len(y)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:] # view

    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()

    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: #(tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def main():
    #train
    #for root, subdirectories, all_files in os.walk(directory_train):
    #for subdirectory in subdirectories:

    all_files = get_list_of_files("train_data")
    files = [f for f in all_files if f.endswith('.wav')]
    distances = np.zeros((len(files), len(files)))
    label = np.zeros(len(files))
    for i in range(len(files)):
        y_train, sr_train = librosa.load(files[i], sr=None)
        mfcc1 = librosa.feature.mfcc(y=y_train, sr=sr_train)
        for j in range(len(files)):
            y_train2, sr_train2 = librosa.load(files[j], sr=None)
            mfcc2 = librosa.feature.mfcc(y=y_train2, sr=sr_train2)
            dist, _, _, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
            distances[i, j] = dist
        start = 'train_data/'
        end = '/'
        s = files[i]
        subdirectory = s[s.find(start) + len(start):s.rfind(end)]
        if(subdirectory == 'dog'):
            label[i] = 0
        elif(subdirectory == 'down'):
            label[i] = 1
        elif(subdirectory == 'off'):
            label[i] = 2
        elif(subdirectory == 'on'):
            label[i] = 3
        elif(subdirectory == 'yes'):
            label[i] = 4
    knn(distances, label, files)

def knn(distances, label, files):
    classifier = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    classifier.fit(distances, label)
    directory = 'test_files'
    preds = []
    for filename in os.listdir(directory):
        y, sr = librosa.load("test_files/" + filename, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        distanceTest = []
        for i in range(len(files)):
            y1, sr1 = librosa.load(files[i])
            mfcc1 = librosa.feature.mfcc(y1, sr1)
            dist, _, _, _ = dtw(mfcc.T, mfcc1.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
            distanceTest.append(dist)
        pred = classifier.predict([distanceTest])[0]
        preds.append(pred)

    #todo add eculidian distance pred and print to file


def get_list_of_files(dir_name):
    list_of_file = os.listdir(dir_name)
    all_files = list()
    for entry in list_of_file:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            all_files.append(full_path)

    return all_files

if __name__ == '__main__':
    main()