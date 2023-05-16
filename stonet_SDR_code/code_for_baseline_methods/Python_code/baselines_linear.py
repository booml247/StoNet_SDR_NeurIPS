import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import PCA
from process_data import preprocess_data
from sliced.datasets import load_athletes
from sliced import SlicedInverseRegression, SlicedAverageVarianceEstimation
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn import metrics
import pandas as pd
import os
import time

data_name = "twonorm"
reduce_dim = 10
mis_rec_sir = []
mis_rec_save = []
mis_rec_pca = []
PCA_time_record = []
for cross_validate_index in range(1):
    # load data
    x_train, y_train, x_test, y_test = preprocess_data(data_name, cross_validate_index)
      
    
    
    if data_name == 'MNIST':
        x_train, x_test = x_train.view(x_train.size(0), -1), x_test.view(x_test.size(0), -1)
        x_train, x_test, y_train, y_test = x_train[:20000], x_test[:20000], y_train[:20000], y_test[:20000]
    
    # fit SIR model
    sir = SlicedInverseRegression(n_directions=reduce_dim, n_slices=11).fit(x_train, y_train)
    x_train_sir = sir.transform(x_train)
    x_test_sir = sir.transform(x_test)
    
    # train a logistic regression model using the dimension reduced data
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train_sir, y_train)
    y_pred = logistic_regression.predict(x_test_sir)
    mis_rate = 1-metrics.accuracy_score(y_test, y_pred)
    print("missclassification rate for SIR is: ", mis_rate)
    mis_rec_sir.append(mis_rate)
    
    # fit SAVE model
    save = SlicedAverageVarianceEstimation(n_directions=reduce_dim, n_slices=11).fit(x_train, y_train)
    x_train_save = save.transform(x_train)
    x_test_save = save.transform(x_test)
    
    # train a logistic regression model using the dimension reduced data
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train_save, y_train)
    y_pred = logistic_regression.predict(x_test_save)
    mis_rate = 1-metrics.accuracy_score(y_test, y_pred)
    print("missclassification rate for SAVE is: ", mis_rate)
    mis_rec_save.append(mis_rate)
    
    # fit PCA
    start_time = time.clock()
    pca = PCA(random_state=123, n_components=reduce_dim).fit(x_train, y_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    time_elapse = time.clock() - start_time
    PCA_time_record.append(time_elapse)
    
    PATH = "./result/"+data_name+ "/PCA/"
    if not os.path.isdir(PATH):
                try:
                    os.makedirs(PATH)
                except OSError as exc:  # Python >2.5
                    if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                        pass
                    else:
                        raise
    f = open(PATH + 'dimension_reduced_data_'+str(reduce_dim)+'.txt', 'wb')
    pickle.dump([x_train_pca, x_test_pca, y_train, y_test], f)
    f.close()
    
    # train a logistic regression model using the dimension reduced data
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train_pca, y_train)
    y_pred = logistic_regression.predict(x_test_pca)
    mis_rate = 1-metrics.accuracy_score(y_test, y_pred)
    print("missclassification rate for PCA is: ", mis_rate)
    mis_rec_pca.append(mis_rate)
    
result = pd.DataFrame({"mis_rec_sir": mis_rec_sir, "mis_rec_save": mis_rec_save, "mis_rec_pca":mis_rec_pca})
import pickle
PATH = "./result/IDA/"+data_name+ "/"
if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise
f = open(PATH + 'result.txt', 'wb')
pickle.dump(result, f)
f.close()
