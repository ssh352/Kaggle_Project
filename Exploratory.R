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
corrplot.mixed(correlations, upper = "square", order = "hclust")

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


## Feature Selection
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(caret))
# str(dm_train, list.len = ncol(dm_train))   ### used to check the complete structure

train <- train %>% select(-id)   ### remove id column
test <- test %>% select(-id)
cat_train <- train[, 1:116]
cat_test <- test[, 1:116]

# dm_train <- model.matrix(loss ~ ., data = train)   ### convert cate to dummy
# dm_test <- model.matrix(~ ., data = test)
# preProc <- preProcess(dm_train, method = "nzv"); preProc   ### get preProcess obj
# dm_train_proc <- predict(preProc, dm_train); dim(dm_train_proc)   ### remove nzv
# dm_test_proc <- predict(preProc, dm_test); dim(dm_test_proc)
# dm_train_proc <- as.data.frame(dm_train_proc)
# dm_test_proc <- as.data.frame(dm_test_proc)


dmcat_train <- model.matrix(~ . + 0, data = cat_train, 
                            contrasts.arg = lapply(cat_train, contrasts, contrasts = FALSE))
dmcat_test <- model.matrix(~ . + 0, data = cat_test, 
                           contrasts.arg = lapply(cat_test, contrasts, contrasts = FALSE))
preProc_obj <- preProcess(dmcat_train, method = "nzv"); preProc_obj
dmcat_train_proc <- predict(preProc_obj, dmcat_train); dim(dmcat_train_proc)
# dmcat_test_proc <- predict(preProc_obj, dmcat_test); dim(dmcat_test_proc)
dmcat_train_proc <- as.data.frame(dmcat_train_proc)
# dmcat_test_proc <- as.data.frame(dmcat_test_proc)

find_same_col <- function(subset, fullset) {
    for (i in colnames(fullset)) {
        if (!(i %in% colnames(subset))) {
            fullset <- fullset[, -which(colnames(fullset) == i)]
        }
    }
    return(fullset)
}

dmcat_test_proc <- find_same_col(dmcat_train_proc, as.data.frame(dmcat_test))

train_processed <- data.frame(dmcat_train_proc, train[, 117:131])
test_processed <- data.frame(dmcat_test_proc, test[, 117:130])




