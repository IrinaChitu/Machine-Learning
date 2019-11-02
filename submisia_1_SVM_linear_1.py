
# coding: utf-8

# In[ ]:

# nr1
SVM linear:
C =  1
predicted: 0.9668888888888889
kaggle: 9.05


# In[11]:

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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


# In[2]:

# loading dataPath | train_files | test_files

dataPath = "D:\\Informatica\\Cursuri Uni\\Anul II\\Semestrul II\\Inteligenta Artificiala - Python\\ML\\PROIECT\\data\\"

train_csvnames = [] # 9000 elem
for filename in os.listdir(dataPath + "train\\"):
    train_csvnames.append(filename)

test_csvnames = [] # 5000 elem
for filename in os.listdir(dataPath + "test\\"):
    test_csvnames.append(filename)
    # print(filename)

# how to: https://www.dataquest.io/blog/numpy-tutorial-python/
with open(dataPath + "train_labels.csv", 'r') as file:
    train_labels_temp = list(csv.reader(file, delimiter=","))

# train_labels
train_labels = np.array(train_labels_temp[1:], dtype=np.int) # headers: id, class

train_labels_user_only = np.zeros(9000)
idx = 0
for label in train_labels:
    train_labels_user_only[idx] = label[1] 
    idx += 1


# In[3]:

# loading ALL training data
print("loading raw train_data")

raw_train_dataset = np.zeros((9000,159,3))
idx = 0
for filename in train_csvnames:
    with open(dataPath + "train\\" + filename, 'r') as file:
        train_data = list(csv.reader(file, delimiter=","))
    if len(train_data) < 159:
        ult_idx = len(train_data)
        for i in range(0, 159-ult_idx):      # unformizam dimensiunile datelor dubland ultima valoare de cate ori este nevoie
            train_data.append(train_data[ult_idx-1])
    train_data = np.array(train_data[:], dtype=np.float64) # headers: id, class
    raw_train_dataset[idx] = train_data
    idx += 1
    
raw_train_dataset = np.array(raw_train_dataset)

print("done loading raw train_data")


# loading ALL testing data
print("loading raw test_data")

raw_test_dataset = np.zeros((5000,159,3)) # [] #
idx = 0
for filename in test_csvnames:
    with open(dataPath + "test\\" + filename, 'r') as file:
        test_data = list(csv.reader(file, delimiter=","))
    if len(test_data) < 159:
        ult_idx = len(test_data)
        for i in range(0, 159-ult_idx):
            test_data.append(test_data[ult_idx-1])
    test_data = np.array(test_data[0:], dtype=np.float64) # headers: id, class
    raw_test_dataset[idx] = test_data
    idx += 1

raw_test_dataset = np.array(raw_test_dataset)

print("done loading raw test_data")


# In[5]:

def get_fft_values(raw_data, T, N, f_s):
    fft_values_ = fft(raw_data)
    fft_values = 2.0 / N * np.abs(fft_values_[:])
    return fft_values


# In[7]:

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
    # fast fourier transform: http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
    # get_fft_values(raw_data, signal_recorded_time / nr_of_values_recorded, nr_of_values_recorded, nr_of_Hz_the_accelerometer_signal_is_recorded_at) 
    fft_column_x = get_fft_values(raw_train_data[:9000, 0], 1.5 / len(column_x), len(column_x), 100)
    fft_column_y = get_fft_values(raw_train_data[:9000, 1], 1.5 / len(column_x), len(column_x), 100)
    fft_column_z = get_fft_values(raw_train_data[:9000, 2], 1.5 / len(column_x), len(column_x), 100)
    
    #raw stats
    feature_for_one_csv.append(stats.mean(column_x))
    # stats.harmonic_mean(column_x)   # exista si valori negative pe care nu stie sa le trateze (sol posibila: aduna o valoare ct)
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
    
    # raw fft
    for i in range(0, len(fft_column_x)):
        feature_for_one_csv.append(fft_column_x[i])
        feature_for_one_csv.append(fft_column_y[i])
        feature_for_one_csv.append(fft_column_z[i])

    #fft stats
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
    
    train_data_features.append(feature_for_one_csv)
    # break
    
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
    # fast fourier transform: http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
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
    
    for i in range(0, len(fft_column_x)):
        feature_for_one_csv.append(fft_column_x[i])
        feature_for_one_csv.append(fft_column_y[i])
        feature_for_one_csv.append(fft_column_z[i])
    
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
    # de adaugat: same ca la train data
   
    test_data_features.append(feature_for_one_csv)

test_data_features = np.array(test_data_features)
print("done")


# In[8]:

def normalize_data(train_data, test_data, type='standard'):
    scaler = preprocessing.StandardScaler()    #  data's distribution will have a mean value 0 and standard deviation of 1
    scaler.fit(train_data)    # expect as input a matrix X with dimensions/shape [number_of_samples, number_of_features]
    train_data_scaled = scaler.transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    return train_data_scaled, test_data_scaled


# In[9]:

# parametrii SVM

svc = svm.SVC()
print(svc)


# In[74]:

# find the best params using GridSearchCV

kernels = ['linear', 'rbf']
Cs = [1e-3, 1e-1, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12]
gammas = [0.001, 0.01, 0.1]

parameters = {
    'kernel': kernels, 
    'C': [1, 3, 5, 7, 10, 12]
#     'gamma': gammas
}

svc = svm.SVC()  # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

clf_svm = GridSearchCV(svc, parameters, cv=5, n_jobs=-1, verbose = 10)

train_dataset_std_svm, test_dataset_std_svm = normalize_data(train_data_features, test_data_features, "standard") # train_dataset_standard, trainlabels, test_dataset_standard, C

clf_svm.fit(train_dataset_std_svm, train_labels_user_only)



# In[75]:

# print the best ones

print(clf_svm.best_params_)
print(clf_svm.best_score_)


# In[56]:

def svm_classifier_linear(train_data, train_labels, test_data, C):
    modelSVM = svm.SVC(C,"linear")
    modelSVM.fit(train_data, train_labels)
    train_labels_predicted = modelSVM.predict(train_data)
    test_labels_predicted = modelSVM.predict(test_data)
    return train_labels_predicted, test_labels_predicted


# In[54]:

# aplly svm classifier with best params found

train_dataset_standard, test_dataset_standard = normalize_data(train_data_features, test_data_features, "standard")

print("applying")

train_labels_predicted, test_labels_predicted = svm_classifier_linear(train_dataset_standard, train_labels_user_only, test_dataset_standard, 1)

print("done applying")


# In[58]:

# format test_labels
test_csvnames = np.array(test_csvnames)
test_labels_predicted_to_send = []
test_labels_predicted_to_send.append(train_labels_temp[0])

idx = 0
for label in test_labels_predicted:
    tag = int(test_csvnames[idx].split('.')[0])
    test_labels_predicted_to_send.append((tag, int(label)))
    idx += 1


# In[59]:

# write csv file to submit
# https://www.blog.pythonlibrary.org/2014/02/26/python-101-reading-and-writing-csv-files/
with open(dataPath + "test_labels 0 9668 - 516 features - stats_raw + fft +fft_stats .csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in test_labels_predicted_to_send:
            print(line)
            writer.writerow(line)
            


# In[9]:

def compute_accuracy(train_labels, predicted_labels):
    return (train_labels==predicted_labels).mean()  


# In[66]:

# prepare cross validation
kfold = KFold(3, True, 1)
accuracy_standard = np.zeros((2, 3))
idx = 0

print("Data shape:")
print("Train: (6000, 516)")
print("Test: (3000, 516)")

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
    train_labels_pred, test_labels_pred = svm_classifier_linear(train_dataset_std, trainlabels, test_dataset_std, C)
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


# In[55]:

# # prepare cross validation
# kfold = KFold(10, True, 1)
# accuracy_standard = np.zeros((2, 10))
# idx = 0

# print("Data shape:")
# print("Train: (8100, 516)")
# print("Test: (900, 516)")

# # split train_dataset
# for train, test in kfold.split(train_data_features):
#     print()
#     if idx == 0:
#         print(idx+1, "st fold")
#     elif idx == 1:
#         print(idx+1, "nd fold")
#     elif idx == 2:
#         print(idx+1, "rd fold")
#     else:
#         print(idx+1, "th fold") 
#     # print('train: %s, test: %s' % (train_dataset[train], train_dataset[test]))
   
#     # get new train_labels and test_labels
#     trainlabels = train_labels_user_only[train]
#     testlabels = train_labels_user_only[test]

#     # normalize data
#     train_dataset_std, test_dataset_std = normalize_data(train_data_features[train], train_data_features[test], "standard") 
#     # apply classifier
#     C = 1
#     train_labels_pred, test_labels_pred = svm_classifier_linear(train_dataset_std, trainlabels, test_dataset_std, C)
#     # calculate accuracy
#     accuracy_standard[0, idx] = compute_accuracy(trainlabels, train_labels_pred)
#     accuracy_standard[1, idx] = compute_accuracy(testlabels, test_labels_pred)
#     print("Accuracy:", accuracy_standard[0, idx], accuracy_standard[1, idx])
    
#     print("Confusion Matrix:")
#     cMatrix = np.zeros((20, 20), dtype=int)
#     for index, predicted_user in enumerate(test_labels_pred):
#     #     print(index, predicted_user, testlabels[index])
#         if predicted_user == testlabels[index]:
#             cMatrix[int(testlabels[index])-1][int(testlabels[index])-1] += 1
#         else:
#             cMatrix[int(testlabels[index])-1][int(predicted_user)-1] += 1 # era i si l am clasificat gresit drept j
#     print(cMatrix)
#     idx += 1

