#change working directory to current dir
wd = getwd()
setwd(wd)
#read data
train <- read.csv('data/train80.csv')[c(-1)] # exclude ID
dim(train)
test <- read.csv('data/test20.csv')[c(-1)] # exclude ID
dim(test)
#str(train)

#Change column diagnosis to factor:
#0 for benign, 1 for malignant
train$diagnosis <- as.factor(train$diagnosis)
#levels(ds$diagnosis) <- c('0','1')
levels(train$diagnosis) <- list("0"="B", "1"="M")

test$diagnosis <- as.factor(test$diagnosis)
levels(test$diagnosis) <- list("0"="B", "1"="M")

## Normalize with UDF
#Custom function for min-max-normalization
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}
#if the data is being normalized we get an ok-ish prediction
#if not it sucks
dataset_norm<-as.data.frame(lapply(train[2:31] ,normalize))
dataset_norm <- cbind(diagnosis=train$diagnosis,dataset_norm)
#str(dataset_norm)
test_norm<-as.data.frame(lapply(test[2:31] ,normalize))
test_norm <- cbind(diagnosis=test$diagnosis,dataset_norm)

library(nnet)
set.seed(42)
#nnet: size  = number of units in the hidden layer.
numUnits = 1
#fit
#model overfitting on trainingdata
#add dropout?
#If the response in formula is a factor, an appropriate classification network is constructed;
#this has one output and entropy fit if the number of levels is two
#note: entropy and softmax are mutually exclusive.
#decay: https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab
NN <-nnet(diagnosis ~. ,
          data = dataset_norm,
          size = numUnits,
          decay = 0.1
);
summary(NN)
#summary(NN$residuals)

prop.table(table(train$diagnosis))

# Predictions on the training set
nnet_predictions_train <-predict(NN, dataset_norm, type = "class")
nnet_predictions_test <-predict(NN, test_norm, type = "class")
#table(test$diagnosis, nnet_predictions_test)

# Confusion matrix on training data
library(caret)
u <- union(nnet_predictions_train, dataset_norm$diagnosis)
t <- table(factor(nnet_predictions_train, u), factor(dataset_norm$diagnosis, u))
confusionMatrix(t)   

u_test <- union(nnet_predictions_test, test$diagnosis)
t_test <- table(factor(nnet_predictions_test, u_test), factor(test$diagnosis, u_test))
confusionMatrix(t_test)   
