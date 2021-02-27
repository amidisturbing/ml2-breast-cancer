#change working directory to current dir
wd = getwd()
setwd(wd)
#read data
train <- read.csv('data/train80.csv')[c(-1)] # exclude ID
dim(train)
#str(train)

#Change column diagnosis to factor:
#0 for benign, 1 for malignant
train$diagnosis <- as.factor(train$diagnosis)
#levels(ds$diagnosis) <- c('0','1')
levels(train$diagnosis) <- list("0"="B", "1"="M")
str(train)

## Normalize with UDF
#Custom function for min-max-normalization
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  str(min, max)
  return (num/denom)
}
dataset_norm<-as.data.frame(lapply(train[2:31] ,normalize))
dataset_norm <- cbind(train$diagnosis,dataset_norm)
#str(dataset_norm)

library(nnet)
#use one layer in order to compare thee activations in the network 
#to simpler approaches
hidden_layers = 10
#fit
NN <-nnet(diagnosis ~. ,
          data= train,
          size=hidden_layers
);
#summary(NN)
#summary(NN$residuals)

prop.table(table(train$diagnosis))

# Predictions on the training set
nnet_predictions_train <-predict(NN, train, type = "class")

# Confusion matrix on training data
library(caret)
u <- union(nnet_predictions_train, train$diagnosis)
t <- table(factor(nnet_predictions_train, u), factor(train$diagnosis, u))
tbl <- table(train$diagnosis, nnet_predictions_train)
tbl
confusionMatrix(t)   
