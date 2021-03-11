#change working directory to current dir
wd = getwd()
setwd(wd)

#libraries
library(corrplot)
library(devtools)
library(caret)
library(funModeling)
library(nnet)
library(caTools)

#read data
ds <- read.csv('../data/train80.csv')[c(-1)] # exclude ID
test <- read.csv('../data/test20.csv')[c(-1)] # exclude ID

#Change column diagnosis to factor:
#0 for benign, 1 for malignant
ds$diagnosis <- as.factor(ds$diagnosis)
levels(ds$diagnosis) <- list("0"="B", "1"="M")

test$diagnosis <- as.factor(test$diagnosis)
levels(test$diagnosis) <- list("0"="B", "1"="M")

#create validation split
# set seed for reproducibility
set.seed(0)
train_rows = sample.split(ds$diagnosis, SplitRatio=0.8)
train = ds[ train_rows,]
validate  = ds[!train_rows,]


# Data Analysis on Training Data
#shouldn't I do this on the whole dataset (as well)?
prop.table(table(ds$diagnosis))
#UNCOMMENT following lines for plots and statistics
#trainingdata_status=df_status(train)
#plot_num(train)

## Normalize with UDF
#Custom function for min-max-normalization
normalize <- function(x) {
  #DEBUG
  #str(x)
  num <- x - min(x)
  #str(min(x),num)
  denom <- max(x) - min(x)
  #DEBUG:
  #str(x)
  #str(min, max) #function (..., na.rm = FALSE)  ???
  return (num/denom)
}
#applies a min/max-normalization to our dataframe of num-values
#and returns it together with it'S factor
apply_normalization <- function(x){
  norm<-as.data.frame(lapply(x[2:31] ,normalize))
  #str(norm)
  norm <- cbind(diagnosis=x$diagnosis,norm)
}
#if the data is being normalized we get an ok-ish prediction
#if not it sucks
train_norm <- apply_normalization(train)
validate_norm <- apply_normalization(validate)
test_norm <- apply_normalization(test)

set.seed(123)
#nnet: size  = number of units in the hidden layer.
numUnits = 10
#fit
#model overfitting on trainingdata
#add dropout?
#If the response in formula is a factor, an appropriate classification network is constructed;
#this has one output and entropy fit if the number of levels is two
#note: entropy and softmax are mutually exclusive.
#decay: https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab
NN <-nnet(diagnosis ~. ,
          data = train_norm
          ,
          size = numUnits
);
#summary(NN)

NN_1 <-nnet(diagnosis ~. ,
            data = train_norm
            ,
            size = numUnits,
            decay = 0.1
);
#summary(NN_1)

NN_skip <-nnet(diagnosis ~. ,
               data = train_norm,
               size = numUnits,
               decay = 0.1,
               skip = TRUE
);
#summary(NN_skip)
#summary(NN$residuals)
model_nnet <- nnet(diagnosis ~ .,
                   data=train,
                   size= 20,
                   decay= 0.01,
                   rang=0.6,
                   trace=TRUE,maxit=200 )

model_nnet_norm <- nnet(diagnosis ~ .,
                        data=train_norm,
                        size= 20,
                        decay= 0.01,
                        rang=0.6,
                        trace=TRUE,maxit=2000 )

prop.table(table(validate$diagnosis))

# Confusion matrix and statistics on training data
evaluate <- function(pred, ref){
  u <- union(pred, ref)
  t <- table(factor(pred, u), factor(ref, u))
  confusionMatrix(t, positive = "1", mode = "everything")
}

# Predictions on the training set
nnet_predictions_train <-predict(NN, train_norm
                                 , type = "class")
evaluate(nnet_predictions_train, train_norm
         $diagnosis)

#evaluate best model on NOT normalized training data
model_nnet_predictions_train <-predict(model_nnet,
                                       train,
                                       type = "class")
evaluate(model_nnet_predictions_train, train
         $diagnosis)

#------------------------------------------------------------------------------
# Evaluation on validation data
#------------------------------------------------------------------------------
#evaluate best model on NOT normalized validation data
model_nnet_predictions_validate <-predict(model_nnet, validate, type = "class")
evaluate(nnet_predictions_validate, validate$diagnosis)

#evaluate best model on normalized validation data - DON'T hold back diagnosis
model_nnet_predictions_validate_norm <-predict(model_nnet, validate_norm, type = "class")
evaluate(model_nnet_predictions_validate_norm, validate_norm$diagnosis)

#evaluate best model trained on normalized data, with NOT normalized data
nnet_predictions_validate_norm <-predict(model_nnet_norm, validate, type = "class")
evaluate(nnet_predictions_validate_norm, validate$diagnosis)

#evaluate best model trained on normalized data, with  normalized data
nnet_predictions_validate_norm <-predict(model_nnet_norm, validate_norm, type = "class")
evaluate(nnet_predictions_validate_norm, validate_norm$diagnosis)
#----------------------------------------------------------------------------------
#further evaluation
nnet_predictions_validate <-predict(NN, validate_norm, type = "class")
evaluate(nnet_predictions_validate, validate$diagnosis)

#best acchieved accuracy for num_of_units = 1 NN_1 ~81% on test data
nnet_predictions_test_1 <-predict(NN_1, validate_norm, type = "class")
evaluate(nnet_predictions_test_1, validate$diagnosis)
#best acchieved accuracy so far for num_of_units = 1 NN_skip ~83% on test data
nnet_predictions_test_skip <-predict(NN_skip, validate_norm, type = "class")
evaluate(nnet_predictions_test_skip, validate$diagnosis)

#------------------------------------------------------------------------------
# Evaluation on test data
#------------------------------------------------------------------------------
#evaluate best model on NOT normalized test data
model_nnet_predictions_test <-predict(model_nnet, test, type = "class")
evaluate(nnet_predictions_test, test$diagnosis)

#evaluate best model on normalized test data
model_nnet_predictions_test_norm <-predict(model_nnet, test_norm, type = "class")
evaluate(model_nnet_predictions_test_norm, test_norm$diagnosis)

#evaluate best model on NOT normalized test data - hold back diagnosis
nnet_predictions_test_norm <-predict(model_nnet_norm, test, type = "class")
evaluate(nnet_predictions_test_norm, test$diagnosis)

#evaluate best model on normalized test data - hold back diagnosis
nnet_predictions_test_norm <-predict(model_nnet_norm, test_norm, type = "class")
evaluate(nnet_predictions_test_norm, test_norm$diagnosis)

#further evaluation
nnet_predictions_test <-predict(NN, test_norm, type = "class")
evaluate(nnet_predictions_test, test$diagnosis)

#best acchieved accuracy for num_of_units = 1 NN_1 ~81% on test data
nnet_predictions_test_1 <-predict(NN_1, test_norm, type = "class")
evaluate(nnet_predictions_test_1, test$diagnosis)
#best acchieved accuracy so far for num_of_units = 1 NN_skip ~83% on test data
nnet_predictions_test_skip <-predict(NN_skip, test_norm, type = "class")
evaluate(nnet_predictions_test_skip, test$diagnosis)

#import the function from Github
#source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

#plot model
#plot.nnet(NN)