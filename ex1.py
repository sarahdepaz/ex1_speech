import numpy as np
import os
import librosa
from scipy.spatial import distance
import re


def dtw(mfcc1, mfcc2):
    dtw_mat = np.zeros([32, 32], np.float64)
    euclid_mat = euclid_mat_helper(mfcc1, mfcc2)
    dtw_mat[0,0] = euclid_mat[0,0]
    for i in range(32):
        for j in range(32):
            if j == 0:
                dtw_mat[i,j] = euclid_mat[i, j] + dtw_mat[i - 1, j]
            elif i == 0:
                dtw_mat[i,j] = euclid_mat[i, j] + dtw_mat[i,j-1]
            elif i > 0 and j > 0:
                dtw_mat[i,j] = euclid_mat[i,j] + min(dtw_mat[i-1, j-1], dtw_mat[i,j-1], dtw_mat[i-1,j])
    return dtw_mat

def euclid_mat_helper(mfcc1, mfcc2):
    euclid_mat = np.zeros([32, 32], np.float32)
    for i, vector1 in enumerate(mfcc1, 0):
        for j, vector2 in enumerate(mfcc2, 0):
            euclid_mat[i,j] = distance.euclidean(vector1,vector2)
    return euclid_mat

# def dtw(x, y, dist):
#     assert len(x)  # Report error while x is none
#     assert len(y)
#     r, c = len(x), len(y)
#     D0 = np.zeros((r + 1, c + 1))
#     D0[0, 1:] = np.inf
#     D0[1:, 0] = np.inf
#     D1 = D0[1:, 1:]  # view
#
#     for i in range(r):
#         for j in range(c):
#             D1[i, j] = dist(x[i], y[j])
#     C = D1.copy()
#
#     for i in range(r):
#         for j in range(c):
#             D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])
#     if len(x) == 1:
#         path = np.zeros(len(y)), range(len(y))
#     elif len(y) == 1:
#         path = range(len(x)), np.zeros(len(x))
#     else:
#         path = _traceback(D0)
#     return D1[-1, -1] / sum(D1.shape), C, D1, path
#
#
# def _traceback(D):
#     i, j = np.array(D.shape) - 2
#     p, q = [i], [j]
#     while (i > 0) or (j > 0):
#         tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
#         if tb == 0:
#             i -= 1
#             j -= 1
#         elif tb == 1:
#             i -= 1
#         else:  #(tb == 2):
#             j -= 1
#         p.insert(0, i)
#         q.insert(0, j)
#     return np.array(p), np.array(q)


def main():
    all_files = get_list_of_files("train_data")
    files = [f for f in all_files if f.endswith('.wav')]
    label = np.zeros(len(files))
    for i in range(len(files)):
        start = 'train_data/'
        end = '/'
        s = files[i]
        subdirectory = s[s.find(start) + len(start):s.rfind(end)]
        if subdirectory == 'dog':
            label[i] = 0
        elif subdirectory == 'down':
            label[i] = 1
        elif subdirectory == 'off':
            label[i] = 2
        elif subdirectory == 'on':
            label[i] = 3
        elif subdirectory == 'yes':
            label[i] = 4
    dtw_preds = knn(label, files)
    euclidian_preds = euclidian_dist(label)
    directory = 'test_files'
    results_to_files = []
    index = 0
    num_to_label = {0.0: 'dog', 1.0: 'down', 2.0: 'off', 3.0: 'on', 4.0: 'yes'}
    for filename in os.listdir(directory):
        results_to_files.append(f"{filename} - {num_to_label[euclidian_preds[index]]} - "
                                f"{num_to_label[dtw_preds[index]]}")
        #results_to_files.append(f"{filename} - {num_to_label[dtw_preds[index]]}")
        index += 1
    with open("output.txt", "w+") as pred:
        pred.write('\n'.join(str(v) for v in results_to_files))
    directory_labeled = "labeled"
    correct = 0
    for root, subdirectories, all_files in os.walk(directory_labeled):
        for subdirectory in subdirectories:
            for filename in os.listdir("labeled/" + subdirectory):
                with open('output.txt') as f:
                    datafile = f.readlines()
                for line in datafile:
                    if filename in line:
                        res = line.split('-')[1]
                        res = re.sub(r"[\n\t\s]*", "", res)
                        if res == subdirectory:
                            correct += 1
    print("\tTest set: Accuracy: {}/{} ({:.0f}%)".format(correct, 250, 100. * correct / 250))

def knn(label, files):
    directory = 'test_files'
    preds = []
    for filename in os.listdir(directory):
        y, sr = librosa.load("test_files/" + filename, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc = librosa.util.normalize(mfcc, 2)
        distance_test = []
        for i in range(len(files)):
            y1, sr1 = librosa.load(files[i], sr=None)
            mfcc1 = librosa.feature.mfcc(y1, sr1)
            mfcc1 = librosa.util.normalize(mfcc1, 2)
            dist = dtw(mfcc.T, mfcc1.T)
            distance_test.append(dist[-1, -1])
        new_min = np.argmin(distance_test)
        pred = label[new_min]
        preds.append(pred)
    return preds



def euclidian_dist(label):
    euclidian_preds = []
    directory = 'test_files'
    all_files = get_list_of_files("train_data")
    files = [f for f in all_files if f.endswith('.wav')]
    for filename in os.listdir(directory):
        results = []
        y, sr = librosa.load("test_files/" + filename, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc = librosa.util.normalize(mfcc, 2)
        for j in range(len(files)):
            y1, sr1 = librosa.load(files[j], sr=None)
            mfcc_train = librosa.feature.mfcc(y=y1, sr=sr1)
            mfcc_train = librosa.util.normalize(mfcc_train, 2)
            avg_dist = vector_euclidian(mfcc, mfcc_train)
            results.append(avg_dist)
        min_euclidian_pred = np.argmin(results)
        pred_euclidian = label[min_euclidian_pred]
        euclidian_preds.append(pred_euclidian)
    return euclidian_preds

def vector_euclidian(mfcc, mfcc_train):
    dist = []
    for i in range(19):
        dist.append(np.linalg.norm(mfcc[i] - mfcc_train[i]))
    return np.average(dist)



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
