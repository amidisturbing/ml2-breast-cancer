#change working directory to current dir
wd = getwd()
setwd(wd)
#read data
train <- read.csv('data/train80.csv')[c(-1)] # exclude ID
dim(train)
str(train)

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
str(dataset_norm)
dataset_norm <- cbind(train$diagnosis,dataset_norm)
colnames(dataset_norm)[1] <- "diagnosis"


library(nnet)
#fit
NN <-nnet(diagnosis ~. ,
          data= dataset_norm,
          size=10
);