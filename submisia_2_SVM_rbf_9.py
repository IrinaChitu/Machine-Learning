
# coding: utf-8

# In[ ]:

# nr2
SVM rbf:
C =  9
predicted: train   |  test
           [1.     0.9165]
kaggle: 0.853


# In[8]:

import numpy as np 
import os
import csv
# https://docs.python.org/3/library/statistics.html#statistics.mode
# https://github.com/python/cpython/blob/3.7/Lib/statistics.py
import statistics as stats
from sklearn import preprocessing
from sklearn import svm
# http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/   +   articol: Person Recognition using Smartphones’ Accelerometer Data
from scipy.fftpack import fft
from sklearn.model_selection import KFold


# In[3]:

# loading dataPath | train_files | test_files

dataPath = "D:\\Informatica\\Cursuri Uni\\Anul II\\Semestrul II\\Inteligenta Artificiala - Python\\ML\\PROIECT\\data\\"

train_csvnames = [] # 9000 elem
for filename in os.listdir(dataPath + "train\\"):
    train_csvnames.append(filename)

test_csvnames = [] # 5000 elem
for filename in os.listdir(dataPath + "test\\"):
    test_csvnames.append(filename)
    # print(filename)

# sursa: https://www.dataquest.io/blog/numpy-tutorial-python/
with open(dataPath + "train_labels.csv", 'r') as file:
    train_labels_temp = list(csv.reader(file, delimiter=","))
# train_labels
train_labels = np.array(train_labels_temp[1:], dtype=np.int) # headers: id, class

train_labels_user_only = np.zeros(9000)
idx = 0
for label in train_labels:
    train_labels_user_only[idx] = label[1] 
    idx += 1


# In[4]:

# (9000,) -> variabil

# loading ALL training data
print("loading raw train_data")

raw_train_dataset = []
for filename in train_csvnames:
    with open(dataPath + "train\\" + filename, 'r') as file:
        train_data = list(csv.reader(file, delimiter=","))
    train_data = np.array(train_data[:], dtype=np.float64) # headers: id, class
    raw_train_dataset.append(train_data)
    
raw_train_dataset = np.array(raw_train_dataset)

print("done loading raw train_data")


# (5000,) -> variabil

# loading ALL testing data
print("loading raw test_data")

raw_test_dataset = []
for filename in test_csvnames:
    with open(dataPath + "test\\" + filename, 'r') as file:
        test_data = list(csv.reader(file, delimiter=","))
    test_data = np.array(test_data[0:], dtype=np.float64) # headers: id, class
    # print(test_data)
    # test_dataset.append(test_data)
    raw_test_dataset.append(test_data)

raw_test_dataset = np.array(raw_test_dataset)

print("done loading raw test_data")


# In[24]:

# min_train = 160
# max_train = 0

# for raw_train_data in raw_train_dataset:
#     train_size = len(raw_train_data)
#     if min_train > train_size:
#         min_train = train_size
#     if max_train < train_size:
#         max_train = train_size
# print(min_train, max_train)
    
# min_test = 160
# max_test = 0
# for raw_test_data in raw_test_dataset:
#     test_size = len(raw_test_data)
#     if min_test > test_size:
#         min_test = test_size
#     if max_test < test_size:
#         max_test = test_size
# print(min_test, max_test)


# In[5]:

def get_fft_values(raw_data, T, N, f_s):
    fft_values_ = fft(raw_data)
    fft_values = 2.0 / N * np.abs(fft_values_[:])
    return fft_values


# In[6]:

# adding train data features
print("adding")

train_data_features = []
# raw_train_data: type = numpy.ndarray   |   shape = ({~150}, 3)
# raw_train_data[0] = un rand din csv    |    raw_train_data[0][0] = x
for raw_train_data in raw_train_dataset:
    feature_for_one_csv = []
    column_x = raw_train_data[:9000, 0]
    column_y = raw_train_data[:9000, 1]
    column_z = raw_train_data[:9000, 2]
    # get_fft_values(raw_data, signal_recorded_time / nr_of_values_recorded, nr_of_values_recorded, nr_of_Hz_the_accelerometer_signal_is_recorded_at) 
    fft_column_x = get_fft_values(raw_train_data[:9000, 0], 1.5 / len(column_x), len(column_x), 100)
    fft_column_y = get_fft_values(raw_train_data[:9000, 1], 1.5 / len(column_x), len(column_x), 100)
    fft_column_z = get_fft_values(raw_train_data[:9000, 2], 1.5 / len(column_x), len(column_x), 100)
    
    feature_for_one_csv.append(stats.mean(column_x))
    # stats.harmonic_mean(column_x)   # exista si valori negative pe care nu stie sa le trateze (sol: poate aduni un anumit numar)
    feature_for_one_csv.append(stats.median(column_x))
    feature_for_one_csv.append(stats.median_low(column_x))
    feature_for_one_csv.append(stats.median_high(column_x))
    feature_for_one_csv.append(stats.median_grouped(column_x))
    # stats.mode(column_x)   # exista 4 equally common values
    feature_for_one_csv.append(stats.pstdev(column_x))
    feature_for_one_csv.append(stats.pvariance(column_x))
    feature_for_one_csv.append(stats.stdev(column_x))
    feature_for_one_csv.append(stats.variance(column_x))
    
    feature_for_one_csv.append(stats.mean(column_y))
    # stats.harmonic_mean(column_y)
    feature_for_one_csv.append(stats.median(column_y))
    feature_for_one_csv.append(stats.median_low(column_y))
    feature_for_one_csv.append(stats.median_high(column_y))
    feature_for_one_csv.append(stats.median_grouped(column_y))
    # stats.mode(column_y)
    feature_for_one_csv.append(stats.pstdev(column_y))
    feature_for_one_csv.append(stats.pvariance(column_y))
    feature_for_one_csv.append(stats.stdev(column_y))
    feature_for_one_csv.append(stats.variance(column_y))
    
    feature_for_one_csv.append(stats.mean(column_z))
    # stats.harmonic_mean(column_z)
    feature_for_one_csv.append(stats.median(column_z))
    feature_for_one_csv.append(stats.median_low(column_z))
    feature_for_one_csv.append(stats.median_high(column_z))
    feature_for_one_csv.append(stats.median_grouped(column_z))
    # stats.mode(column_z)
    feature_for_one_csv.append(stats.pstdev(column_z))
    feature_for_one_csv.append(stats.pvariance(column_z))
    feature_for_one_csv.append(stats.stdev(column_z))
    feature_for_one_csv.append(stats.variance(column_z))
    
    feature_for_one_csv.append(stats.mean(fft_column_x))
    feature_for_one_csv.append(stats.mean(fft_column_y))
    feature_for_one_csv.append(stats.mean(fft_column_z))
    feature_for_one_csv.append(stats.median_low(fft_column_x))
    feature_for_one_csv.append(stats.median_low(fft_column_y))
    feature_for_one_csv.append(stats.median_low(fft_column_z))
    feature_for_one_csv.append(stats.median_high(fft_column_x))
    feature_for_one_csv.append(stats.median_high(fft_column_y))
    feature_for_one_csv.append(stats.median_high(fft_column_z))
    feature_for_one_csv.append(stats.median_grouped(fft_column_x))
    feature_for_one_csv.append(stats.median_grouped(fft_column_y))
    feature_for_one_csv.append(stats.median_grouped(fft_column_z))
    # mai adauga din articole ce mai gasesti
    # cupleaza cate 42 (pt ca mean vrea cel putin un punct si stdev vrea 2) datele si calculeaza aceleasi lucruri ca mai sus
    idx = 0
    dim = int(len(column_x) / 40)
    rest = len(column_x) % 40
    for calut in range(0, 40):
        if rest != 0:
            pas = dim + 1
        else:
            pas = dim
        grup_x = column_x[idx:idx+dim]
        grup_y = column_y[idx:idx+dim]
        grup_z = column_z[idx:idx+dim]
        idx += dim
        
        feature_for_one_csv.append(stats.mean(grup_x))
        # feature_for_one_csv.append(stats.harmonic_mean(grup_x))
        feature_for_one_csv.append(stats.median(grup_x))
        feature_for_one_csv.append(stats.median_low(grup_x))
        feature_for_one_csv.append(stats.median_high(grup_x))
        feature_for_one_csv.append(stats.median_grouped(grup_x))
        # feature_for_one_csv.append(stats.mode(grup_x))
        feature_for_one_csv.append(stats.pstdev(grup_x))
        feature_for_one_csv.append(stats.pvariance(grup_x))
        feature_for_one_csv.append(stats.stdev(grup_x))
        feature_for_one_csv.append(stats.variance(grup_x))
        
        feature_for_one_csv.append(stats.mean(grup_y))
        # feature_for_one_csv.append(stats.harmonic_mean(grup_y))
        feature_for_one_csv.append(stats.median(grup_y))
        feature_for_one_csv.append(stats.median_low(grup_y))
        feature_for_one_csv.append(stats.median_high(grup_y))
        feature_for_one_csv.append(stats.median_grouped(grup_y))
        # feature_for_one_csv.append(stats.mode(grup_y))
        feature_for_one_csv.append(stats.pstdev(grup_y))
        feature_for_one_csv.append(stats.pvariance(grup_y))
        feature_for_one_csv.append(stats.stdev(grup_y))
        feature_for_one_csv.append(stats.variance(grup_y))
        
        feature_for_one_csv.append(stats.mean(grup_z))
        # feature_for_one_csv.append(stats.harmonic_mean(grup_z))
        feature_for_one_csv.append(stats.median(grup_z))
        feature_for_one_csv.append(stats.median_low(grup_z))
        feature_for_one_csv.append(stats.median_high(grup_z))
        feature_for_one_csv.append(stats.median_grouped(grup_z))
        # feature_for_one_csv.append(stats.mode(grup_z))
        feature_for_one_csv.append(stats.pstdev(grup_z))
        feature_for_one_csv.append(stats.pvariance(grup_z))
        feature_for_one_csv.append(stats.stdev(grup_z))
        feature_for_one_csv.append(stats.variance(grup_z))

    train_data_features.append(feature_for_one_csv)
    
train_data_features = np.array(train_data_features)
print("done")


# In[7]:

# adding test data features
print("adding")

test_data_features = []
# raw_test_data: type = numpy.ndarray   |   shape = ({~150}, 3)
# raw_test_data[0] = un rand din csv    |    raw_test_data[0][0] = x
for raw_test_data in raw_test_dataset:
    feature_for_one_csv = []
    column_x = raw_test_data[:9000, 0]
    column_y = raw_test_data[:9000, 1]
    column_z = raw_test_data[:9000, 2]
    fft_column_x = get_fft_values(raw_test_data[:9000, 0], 1.5 / len(column_x), len(column_x), 100)
    fft_column_y = get_fft_values(raw_test_data[:9000, 1], 1.5 / len(column_x), len(column_x), 100)
    fft_column_z = get_fft_values(raw_test_data[:9000, 2], 1.5 / len(column_x), len(column_x), 100)
    
    feature_for_one_csv.append(stats.mean(column_x))
    # stats.harmonic_mean(column_x)   # exista si valori negative pe care nu stie sa le trateze (sol: poate aduni un anumit numar)
    feature_for_one_csv.append(stats.median(column_x))
    feature_for_one_csv.append(stats.median_low(column_x))
    feature_for_one_csv.append(stats.median_high(column_x))
    feature_for_one_csv.append(stats.median_grouped(column_x))
    # stats.mode(column_x)   # exista 4 equally common values
    feature_for_one_csv.append(stats.pstdev(column_x))
    feature_for_one_csv.append(stats.pvariance(column_x))
    feature_for_one_csv.append(stats.stdev(column_x))
    feature_for_one_csv.append(stats.variance(column_x))
    
    feature_for_one_csv.append(stats.mean(column_y))
    # stats.harmonic_mean(column_y)
    feature_for_one_csv.append(stats.median(column_y))
    feature_for_one_csv.append(stats.median_low(column_y))
    feature_for_one_csv.append(stats.median_high(column_y))
    feature_for_one_csv.append(stats.median_grouped(column_y))
    # stats.mode(column_y)
    feature_for_one_csv.append(stats.pstdev(column_y))
    feature_for_one_csv.append(stats.pvariance(column_y))
    feature_for_one_csv.append(stats.stdev(column_y))
    feature_for_one_csv.append(stats.variance(column_y))
    
    feature_for_one_csv.append(stats.mean(column_z))
    # stats.harmonic_mean(column_z)
    feature_for_one_csv.append(stats.median(column_z))
    feature_for_one_csv.append(stats.median_low(column_z))
    feature_for_one_csv.append(stats.median_high(column_z))
    feature_for_one_csv.append(stats.median_grouped(column_z))
    # stats.mode(column_z)
    feature_for_one_csv.append(stats.pstdev(column_z))
    feature_for_one_csv.append(stats.pvariance(column_z))
    feature_for_one_csv.append(stats.stdev(column_z))
    feature_for_one_csv.append(stats.variance(column_z))
    
    feature_for_one_csv.append(stats.mean(fft_column_x))
    feature_for_one_csv.append(stats.mean(fft_column_y))
    feature_for_one_csv.append(stats.mean(fft_column_z))
    feature_for_one_csv.append(stats.median_low(fft_column_x))
    feature_for_one_csv.append(stats.median_low(fft_column_y))
    feature_for_one_csv.append(stats.median_low(fft_column_z))
    feature_for_one_csv.append(stats.median_high(fft_column_x))
    feature_for_one_csv.append(stats.median_high(fft_column_y))
    feature_for_one_csv.append(stats.median_high(fft_column_z))
    feature_for_one_csv.append(stats.median_grouped(fft_column_x))
    feature_for_one_csv.append(stats.median_grouped(fft_column_y))
    feature_for_one_csv.append(stats.median_grouped(fft_column_z))
    idx = 0
    dim = int(len(column_x) / 40)
    rest = len(column_x) % 40
    for calut in range(0, 40):
        if rest != 0:
            pas = dim + 1
        else:
            pas = dim
        grup_x = column_x[idx:idx+dim]
        grup_y = column_y[idx:idx+dim]
        grup_z = column_z[idx:idx+dim]
        idx += dim
        
        feature_for_one_csv.append(stats.mean(grup_x))
        # feature_for_one_csv.append(stats.harmonic_mean(grup_x))
        feature_for_one_csv.append(stats.median(grup_x))
        feature_for_one_csv.append(stats.median_low(grup_x))
        feature_for_one_csv.append(stats.median_high(grup_x))
        feature_for_one_csv.append(stats.median_grouped(grup_x))
        # feature_for_one_csv.append(stats.mode(grup_x))
        feature_for_one_csv.append(stats.pstdev(grup_x))
        feature_for_one_csv.append(stats.pvariance(grup_x))
        feature_for_one_csv.append(stats.stdev(grup_x))
        feature_for_one_csv.append(stats.variance(grup_x))
        
        feature_for_one_csv.append(stats.mean(grup_y))
        # feature_for_one_csv.append(stats.harmonic_mean(grup_y))
        feature_for_one_csv.append(stats.median(grup_y))
        feature_for_one_csv.append(stats.median_low(grup_y))
        feature_for_one_csv.append(stats.median_high(grup_y))
        feature_for_one_csv.append(stats.median_grouped(grup_y))
        # feature_for_one_csv.append(stats.mode(grup_y))
        feature_for_one_csv.append(stats.pstdev(grup_y))
        feature_for_one_csv.append(stats.pvariance(grup_y))
        feature_for_one_csv.append(stats.stdev(grup_y))
        feature_for_one_csv.append(stats.variance(grup_y))
        
        feature_for_one_csv.append(stats.mean(grup_z))
        # feature_for_one_csv.append(stats.harmonic_mean(grup_z))
        feature_for_one_csv.append(stats.median(grup_z))
        feature_for_one_csv.append(stats.median_low(grup_z))
        feature_for_one_csv.append(stats.median_high(grup_z))
        feature_for_one_csv.append(stats.median_grouped(grup_z))
        # feature_for_one_csv.append(stats.mode(grup_z))
        feature_for_one_csv.append(stats.pstdev(grup_z))
        feature_for_one_csv.append(stats.pvariance(grup_z))
        feature_for_one_csv.append(stats.stdev(grup_z))
        feature_for_one_csv.append(stats.variance(grup_z))
        
        # print(feature_for_one_csv)
    test_data_features.append(feature_for_one_csv)

test_data_features = np.array(test_data_features)
print("done")


# In[9]:

def normalize_data(train_data, test_data, type=None):
    if type == None:
        return train_data, test_data
    if type == 'standard':
        scaler = preprocessing.StandardScaler()    #  data's distribution will have a mean value 0 and standard deviation of 1
        scaler.fit(train_data)    # expect as input a matrix X with dimensions/shape [number_of_samples, number_of_features]
        train_data_scaled = scaler.transform(train_data)
        test_data_scaled = scaler.transform(test_data)
        return train_data_scaled, test_data_scaled
    if type == "min_max":  # scalare 0-1
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        min_max_scaler.fit(x_train)
        train_data_scaled = min_max_scaler.transform(train_data)
        test_data_scaled = min_max_scaler.transform(test_data)
        return train_data_scaled, test_data_scaled
    if type == 'L1':
        train_data_l1 = preprocessing.normalize(train_data, norm="l1", axis=1)
        test_data_l1 = preprocessing.normalize(test_data, norm="l1", axis=1)
        return train_data_l1, test_data_l1
    if type == 'L2':
        train_data_l2 = preprocessing.normalize(train_data, norm="l2", axis=1)
        test_data_l2 = preprocessing.normalize(test_data, norm="l2", axis=1)
        return train_data_l2, test_data_l2


# In[10]:

def svm_classifier(train_data, train_labels, test_data, C):
    modelSVM = svm.SVC(C,"rbf")
    modelSVM.fit(train_data, train_labels)
    train_labels_predicted = modelSVM.predict(train_data)
    test_labels_predicted = modelSVM.predict(test_data)
    return train_labels_predicted, test_labels_predicted


# In[13]:

# find classifier

# split training data to simulate testing data
train = train_data_features[:7000]
# print(train)
test = train_data_features[7000:]
# print(test)

# verificare
# print(train_data_features[0])
# print(train_data_features[7000])
# print(train[0])
# print(test[0])

# split labels
trainlabels = train_labels_user_only[:7000]
testlabels = train_labels_user_only[7000:]

# Cs = [2, 3, 4, 5, 6, 7, 8, 9, 11]
Cs = [6, 7, 8, 9, 11]

# C mic => accent pe margine mare
# C mare => pune accent pe clasificare perfecta

accuracy_standard = np.zeros((2, len(Cs)))
train_dataset_standard, test_dataset_standard = normalize_data(train, test, "standard")
for i in range(len(Cs)):
    C = Cs[i]
    print("C = ", C)
    train_labels_predicted, test_labels_predicted = svm_classifier(train_dataset_standard, trainlabels, test_dataset_standard, C)
    accuracy_standard[0, i] = compute_accuracy(trainlabels, train_labels_predicted)
    accuracy_standard[1, i] = compute_accuracy(testlabels, test_labels_predicted)
    print(accuracy_standard[:,i])
# print(accuracy_standard)


# In[11]:

# apply svm classifier rbf C = 9

# normalizing data
train_dataset_standard, test_dataset_standard = normalize_data(train_data_features, test_data_features, "standard")

print("applying")

C = 9
train_labels_predicted, test_labels_predicted = svm_classifier(train_dataset_standard, train_labels_user_only, test_dataset_standard, C)

print("done applying")


# In[22]:

# format test_labels
test_csvnames = np.array(test_csvnames)
test_labels_predicted_to_send = []
test_labels_predicted_to_send.append(train_labels_temp[0])

idx = 0
for label in test_labels_predicted:
    tag = int(test_csvnames[idx].split('.')[0])
    test_labels_predicted_to_send.append((tag, int(label)))
    idx += 1


# In[23]:

# write csv file to submit
# https://www.blog.pythonlibrary.org/2014/02/26/python-101-reading-and-writing-csv-files/
with open(dataPath + "test_labels - final_2 - rbf - C = 9 - 1119 features.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in test_labels_predicted_to_send:
#             line = np.array(line, dtype=np.int32)
            print(line)
            writer.writerow(line)
            


# In[12]:

def compute_accuracy(train_labels, predicted_labels):
    return (train_labels==predicted_labels).mean()  


# In[16]:

# prepare cross validation
kfold = KFold(3, True, 1)
accuracy_standard = np.zeros((2, 3))
idx = 0

print("Data shape:")
print("Train: (6000, 1119)")
print("Test: (3000, 1119)")

# split train_dataset
for train, test in kfold.split(train_data_features):
    print()
    if idx == 0:
        print(idx+1, "st fold")
    elif idx == 1:
        print(idx+1, "nd fold")
    elif idx == 2:
        print(idx+1, "rd fold")
    # print('train: %s, test: %s' % (train_data_features[train], train_data_features[test]))
    # print(train_data_features[train].shape, train_data_features[test].shape)
    # get new train_labels and test_labels
    trainlabels = train_labels_user_only[train]
    testlabels = train_labels_user_only[test]

    # normalize data
    train_dataset_std, test_dataset_std = normalize_data(train_data_features[train], train_data_features[test], "standard") 
    # apply classifier
    C = 1
    train_labels_pred, test_labels_pred = svm_classifier(train_dataset_std, trainlabels, test_dataset_std, C)
    # calculate accuracy
    accuracy_standard[0, idx] = compute_accuracy(trainlabels, train_labels_pred)
    accuracy_standard[1, idx] = compute_accuracy(testlabels, test_labels_pred)
    print("Accuracy:", accuracy_standard[0, idx], accuracy_standard[1, idx])
    
    print("Confusion Matrix:")
    cMatrix = np.zeros((20, 20), dtype=int)
    for index, predicted_user in enumerate(test_labels_pred):
    #     print(index, predicted_user, testlabels[index])
        if predicted_user == testlabels[index]:
            cMatrix[int(testlabels[index])-1][int(testlabels[index])-1] += 1
        else:
            cMatrix[int(testlabels[index])-1][int(predicted_user)-1] += 1 # era i si l am clasificat gresit drept j
    print(cMatrix)
    idx += 1

print("The mean accuracy rate for a 3-fold cross-validation procedure on the training set: ", accuracy_standard[1].mean())

