## import libraries
import numpy as np
np.random.seed(123)
​
import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
​
## Batch generators #####################################################################################
​
def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0
​
def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0
​
#########################################################################################################
​
## read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
​
## set test loss to NaN
test['loss'] = np.nan
​
## response and IDs
y = train['loss'].values
id_train = train['id'].values
id_test = test['id'].values
​
## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)
​
## Preprocessing and transforming to sparse data
sparse_data = []
​
f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)
​
f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)
​
del(tr_te, train, test)
​
## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]
​
print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)
​
del(xtr_te, sparse_data, tmp)
​
## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)
​
## cv-folds
nfolds = 5
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)
​
## train models
i = 0
nbags = 5
nepochs = 55
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])
​
for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  verbose = 0)
        pred += model.predict_generator(generator = batch_generatorp(xte, 800, False), 
                                        val_samples = xte.shape[0])[:,0]
        pred_test += model.predict_generator(generator = batch_generatorp(xtest, 800, False), 
                                             val_samples = xtest.shape[0])[:,0]
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(yte, pred)
    i += 1
    print('Fold ', i, '- MAE:', score)
​
print('Total - MAE:', mean_absolute_error(y, pred_oob))
​
## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv('preds_oob.csv', index = False)
​
## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('submission_keras.csv', index = False)
Using TensorFlow backend.
('Dim train', (188318, 1190))
('Dim test', (125546, 1190))
/Users/David/anaconda/lib/python2.7/site-packages/keras/engine/training.py:1480: UserWarning: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.
  warnings.warn('Epoch comprised more than '
('Fold ', 1, '- MAE:', 1128.8422538324971)
('Fold ', 2, '- MAE:', 1135.935483373449)
('Fold ', 3, '- MAE:', 1150.960146186135)
('Fold ', 4, '- MAE:', 1134.7918331547392)
('Fold ', 5, '- MAE:', 1140.6153925382982)
('Total - MAE:', 1138.2290273970423)
