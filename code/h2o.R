#change working directory to current dir
wd = getwd()
setwd(wd)
#install.packages("h2o")

library(h2o)

#start a local h2o cluster:
localH2O = h2o.init(ip="localhost", port = 5321, 
                    startH2O = TRUE, nthreads=-1)
h2o.removeAll() ## clean slate - just in case the cluster was already running

#load data with h2o and create stratified cross validation split
test <- h2o.importFile('../data/test20.csv')[c(-1)]

#Change column diagnosis to factor:
#0 for benign, 1 for malignant
test $diagnosis <- as.factor(test$diagnosis)
levels(test$diagnosis) <- list("0"="B", "1"="M")

#load data with h2o and create stratified cross validation split
train <- h2o.importFile('../data/train80.csv')[c(-1)]

#Change column diagnosis to factor:
#0 for benign, 1 for malignant
train$diagnosis <- as.factor(train$diagnosis)
levels(train$diagnosis) <- list("0"="B", "1"="M")

## Model Parameters
target = names(train)[1]
predictors = names(train)[2:31]
# try using the fold_assignment parameter:
# note you must set nfolds to use this parameter
assignment_type <- 'Stratified'
activation_opt <- c("TanhWithDropout", "RectifierWithDropout", "MaxoutWithDropout")

#Preprocessing is done implicitly by h2o
#before even training your net it computes mean and standard deviation
#of all the features you have, and replaces the original values with (x - mean) / stddev.
#src: https://stackoverflow.com/questions/37798134/h2o-deep-learning-weights-and-normalization
#Set timer:
timer <- proc.time()

#Set Grid parameters:
hidden_opt <- list(c(32,32,32), c(5,25,75), c(100,100,100))
l1_opt     <- c(1e-5, 1e-4,1e-3)
hidden_drpoutRatios <- list(c(0.5,0.5,0.5), c(0.5,0.3,0.2), c(0.1,0.2,0.8))
hyper_pars <- list(hidden = hidden_opt, hidden_dropout_ratios = hidden_drpoutRatios, l1 = l1_opt, activation = activation_opt)
#Building Grid models:
model_grid <- h2o.grid(
  algorithm = "deeplearning",
  hyper_params = hyper_pars,
  x = predictors,
  y = target,
  training_frame = train,
  fold_assignment = assignment_type,
  input_dropout_ratio = 0.2,
  balance_classes = T,
  momentum_stable = 0.99,
  nesterov_accelerated_gradient = T,
  epochs = 500,
  nfolds = 10,
  variable_importances = T,
  keep_cross_validation_predictions = T)

for (model_id in model_grid@model_ids) {
  current_model <- h2o.getModel(model_id)

  auc <- h2o.auc(h2o.getModel(model_id))
  # aucPR (Area Under PRECISION RECALL Curve)
  aucpr <- h2o.aucpr(h2o.getModel(model_id))
  #performance <-  h2o.performance(h2o.getModel(model_id))
  print(model_id)
  print(sprintf('CV set auc: %f', auc))
  print(aucpr)
  #print(performance)
  print("-----------------------------------")
}
#preparing best model
best_model = h2o.getModel("Grid_DeepLearning_RTMP_sid_a746_15_model_R_1615916806869_10538_model_15")
#save model to file
model_path <- h2o.saveModel(object = best_model, path = getwd(), force = TRUE)
# load the model
model_path = "../models/Grid_DeepLearning_RTMP_sid_a746_15_model_R_1615916806869_10538_model_15"
saved_model <- h2o.loadModel(model_path)
#best_model = saved_model

# Get fitted values of breast cancer dataset
cancer.fit = h2o.predict(object = best_model, newdata = test)
summary(cancer.fit)
# save the model
# the model was choosen using cross-validation
# we've picked the model with the highest AUC as well as aucPR, where the accievend value was still BELOW 1 (overfitting).
# model_path <- h2o.saveModel(object = current_model, path = getwd(), force = TRUE)
# print(model_path)

h2o.performance(best_model, test)
h2o.varimp_plot(best_model)
# Create the partial dependence plot
h2o.pd_plot(best_model, test, column = 'area_se')
h2o.ice_plot(best_model, test, column = 'smoothness_worst')
#shut down H2O instance
#h2o.shutdown()