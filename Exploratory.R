# load the training data
setwd("C:/Users/Xinyuan Wu/Desktop/Xinyuan's Repo/Kaggle_Project")
train <- read.csv("data/train.csv/train.csv")
test <- read.csv("data/test.csv/test.csv")
num_train <- train[, -c(1: 117, 132)]
num_test <- test[, -c(1:117)]
str(num_train); str(num_test)

## correlation plot
suppressPackageStartupMessages(library(corrplot))
correlations <- cor(num_train)
corrplot(correlations, method = "square", order = "hclust")

## explore continuous variables
draw_cont <- function(data, col) {
    len <- sapply(data, function(x) length(unique(x)))
    plot(1:len[col], sort(unique(data[, col]), decreasing = FALSE),
         xlab = paste0('Cont', col), ylab = "Value")
}

par(mfrow = c(3, 5))
for (i in 1:14) {
    draw_cont(num_train, col = i)
}

par(mfrow = c(3, 5))
for (i in 1:7) {
    draw_cont(num_train, col = i)
    draw_cont(num_test, col = i)
}

par(mfrow = c(3, 5))
for (i in 8:14) {
    draw_cont(num_train, col = i)
    draw_cont(num_test, col = i)
}

## check level consistency between train and test
compare_level <- function(data1, data2) {
    
}

## back-transformation of cont variables
back_trans <- function(data) {
    for (i in 1:dim(data)[2]) {
        a <- factor(data[, i], 
                    levels = sort(unique(data[, i]), decreasing = FALSE))
        levels(a) <- 1:length(unique(data[, i]))
        data[, i] <- as.numeric(as.character(a))
    }
    return(data)
}

num_train_trans <- back_trans(num_train)
num_test_trans <- back_trans(num_test)





















