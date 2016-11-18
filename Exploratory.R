# load the training data
setwd("C:/Users/Xinyuan Wu/Desktop/Xinyuan's Repo/Kaggle_Project")
train <- read.csv("data/train.csv/train.csv")
num <- train[, -c(1: 117, 132)]

## correlation plot
suppressPackageStartupMessages(library(corrplot))
correlations <- cor(num)
corrplot(correlations, method = "square", order="hclust")

