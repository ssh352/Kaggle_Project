# load the training data
setwd("C:/Users/Xinyuan Wu/Desktop/Xinyuan's Repo/Kaggle_Project")
train <- read.csv("data/train.csv/train.csv")
train_encoded <- read.csv("data/train.csv/train_encoded.csv")
train_encoded$loss <- train[, 'loss']

# extract numeric variables
num <- train[, -c(1: 117, 132)]
num_scale <- scale(num)

# test colmeans and sd for each column 
colMeans(num_scale); apply(num_scale, 2, sd)

# perform svd
num_scale_svd <- svd(num_scale)

# replace numeric variable with svd result
train[, c(118, 131)] <- num_scale_svd$u

# try rf
library(randomForest)
rf1 = randomForest(loss ~ ., data = train_encoded, mtry = 50)
str(train_encoded, list.len = ncol(train))

# rf with only numeric
