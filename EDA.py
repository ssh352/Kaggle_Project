import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

print "Start"

dat_train = pd.read_csv("train.csv")
dat_test = pd.read_csv("test.csv")

ID = dat_test['id']

dat_test.drop('id', axis=1, inplace=True)
dat_train = dat_train.iloc[:, 1:]

# print dat_test.head(5)
# print dat_train.head(5)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

labels = []
split = 116
cols = dat_train.columns



for i in range(0,split):
    train = dat_train[cols[i]].unique()
    test = dat_test[cols[i]].unique()
    labels.append(list(set(train) | set(test)))
    # print "train"
    # print train
    # print "test"
    # print test
    # print "labels"
    # print labels

cats = []
for i in range(0, split):
    # print labels[i]
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dat_train.iloc[:,i])
    # print feature
    feature = feature.reshape(dat_train.shape[0], 1)
    # print feature
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    # print feature
    cats.append(feature)

encoded_cats = np.column_stack(cats)
dataset_encoded = np.concatenate((encoded_cats,dat_train.iloc[:,split:].values),axis=1)
print "finished encoding"
print dataset_encoded
print dataset_encoded.shape
# pd.DataFrame(dataset_encoded).to_csv("train_ONE_raw.csv", index=False)
