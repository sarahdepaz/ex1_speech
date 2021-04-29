import numpy as np
import os
import librosa
from scipy.spatial import distance
import re


def dtw(mfcc1, mfcc2):
    dtw_mat = np.zeros([32, 32], np.float64)
    euclid_mat = euclid_mat_helper(mfcc1, mfcc2)
    dtw_mat[0, 0] = euclid_mat[0, 0]
    for i in range(32):
        for j in range(32):
            if j == 0:
                dtw_mat[i, j] = euclid_mat[i, j] + dtw_mat[i - 1, j]
            elif i == 0:
                dtw_mat[i, j] = euclid_mat[i, j] + dtw_mat[i, j - 1]
            elif i > 0 and j > 0:
                dtw_mat[i, j] = euclid_mat[i, j] + min(dtw_mat[i - 1, j - 1], dtw_mat[i, j - 1], dtw_mat[i - 1, j])
    return dtw_mat


def euclid_mat_helper(mfcc1, mfcc2):
    euclid_mat = np.zeros([32, 32], np.float32)
    for i, vector1 in enumerate(mfcc1, 0):
        for j, vector2 in enumerate(mfcc2, 0):
            euclid_mat[i, j] = distance.euclidean(vector1, vector2)
    return euclid_mat


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
        # results_to_files.append(f"{filename} - {num_to_label[dtw_preds[index]]}")
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
                        res = line.split('-')[2]
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
        mfcc = normalize(mfcc.T)
        distance_test = []
        for i in range(len(files)):
            y1, sr1 = librosa.load(files[i], sr=None)
            mfcc1 = librosa.feature.mfcc(y1, sr1)
            mfcc1 = normalize(mfcc1.T)
            dist = dtw(mfcc.T, mfcc1.T)
            distance_test.append(dist[-1, -1])
        new_min = np.argmin(distance_test)
        pred = label[new_min]
        preds.append(pred)
    return preds

def normalize(matrix):
    for i in range(32):
        avg = np.mean(matrix[i])
        matrix[i] = matrix[i] - avg
        std = np.std(matrix[i])
        matrix[i] = np.divide(matrix[i], std)
    return matrix.T

def normalize_minmax(data):
    normalized_data = []
    for col in data.T:
        max_val = np.max(col)
        min_val = np.min(col)
        rng = max_val - min_val
        normalized_col = np.divide(col-min_val, rng)
        normalized_data.append(normalized_col)
    return np.array(normalized_data).T

def euclidian_dist(label):
    euclidian_preds = []
    directory = 'test_files'
    all_files = get_list_of_files("train_data")
    files = [f for f in all_files if f.endswith('.wav')]
    for filename in os.listdir(directory):
        results = []
        y, sr = librosa.load("test_files/" + filename, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc = normalize(mfcc.T)
        for j in range(len(files)):
            y1, sr1 = librosa.load(files[j], sr=None)
            mfcc_train = librosa.feature.mfcc(y=y1, sr=sr1)
            mfcc_train = normalize(mfcc_train.T)
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
