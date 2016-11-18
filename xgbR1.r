library(data.table)
library(Matrix)
library(xgboost)
library(Metrics)

ID = 'id'
TARGET = 'loss'
SEED = 1314

setwd("C:/Users/Xinyuan Wu/Desktop/Xinyuan's Repo/Kaggle_Project")
TRAIN_FILE = "data/train.csv/train.csv"
TEST_FILE = "data/test.csv/test.csv"
SUBMISSION_FILE = "sample_submission.csv"


train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

y_train = log(train[,TARGET, with = FALSE])[[TARGET]]

train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]

ntrain = nrow(train)
train_test = rbind(train, test)

features = names(train)

for (f in features) {
  if (class(train_test[[f]])=="character") {
    #cat("VARIABLE : ",f,"\n")
    levels <- unique(train_test[[f]])
    train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
  }
}


x_train = train_test[1:ntrain,]
x_test = train_test[(ntrain+1):nrow(train_test),]


dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dtest = xgb.DMatrix(as.matrix(x_test))


xgb_params = list(
  seed = 1314,
  colsample_bytree = 0.7,
  subsample = 0.7,
  eta = 0.075,
  objective = 'reg:linear',
  max_depth = 6,
  num_parallel_tree = 1,
  min_child_weight = 1,
  base_score = 7
)

xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

res = xgb.cv(xgb_params,
             dtrain,
             nrounds=750,
             nfold=4,
             early_stopping_rounds=15,
             print_every_n = 10,
             verbose= 1,
             feval=xg_eval_mae,
             maximize=FALSE)

best_nrounds = res$best_iteration
cv_mean = res$evaluation_log$test_error_mean[best_nrounds]
cv_std = res$evaluation_log$test_error_std[best_nrounds]
cat(paste0('CV-Mean: ',cv_mean,' ', cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = fread(SUBMISSION_FILE, colClasses = c("integer", "numeric"))
submission$loss = exp(predict(gbdt,dtest))
write.csv(submission,'xgb_starter_v2.hehehe.csv',row.names = FALSE)

#1126.19439 on PLB