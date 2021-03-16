#change working directory to current dir
wd = getwd()
setwd(wd)
#install.packages("h2o")
library(caTools)

#read data
ds <- read.csv('../data/train80.csv')[c(-1)] # exclude ID
library(h2o)

#Change column diagnosis to factor:
#0 for benign, 1 for malignant
ds$diagnosis <- as.factor(ds$diagnosis)
levels(ds$diagnosis) <- list("0"="B", "1"="M")

#create validation split
# set seed for reproducibility
set.seed(2)
train_rows = sample.split(ds$diagnosis, SplitRatio=0.8)
train = ds[ train_rows,]
validate  = ds[!train_rows,]

#TODO:Add Crossvalidation: 
#stratified:
#http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/fold_assignment.html?highlight=stratified#example

#start a local h2o cluster:
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train_h2o <- as.h2o(train)
validate_h2o <- as.h2o(validate)

#load data with h2oand create stratified cross validation split
data <- h2o.importFile('../data/train80.csv')[c(-1)]

#Change column diagnosis to factor:
#0 for benign, 1 for malignant
data$diagnosis <- as.factor(data$diagnosis)
levels(data$diagnosis) <- list("0"="B", "1"="M")

# try using the fold_assignment parameter:
# note you must set nfolds to use this parameter
assignment_type <- 'Stratified'

target = names(train_h2o)[1]
predictors = names(train_h2o)[2:31]

model <-  h2o.deeplearning(model_id = 'model_cv',
                           x = 2:31,
                           y = target,
                           training_frame = data,
                           fold_assignment = assignment_type,
                           distribution = "multinomial",
                           epochs = 4,
                           activation = 'RectifierWithDropout',
                           nfolds = 5,
                           seed = 1234)
# Build the first deep learning model, specifying the model_id so you
# can indicate which model to use when you want to continue training.
# We will use 4 epochs to start off with and then build an additional
# 16 epochs with checkpointing.
dl <- h2o.deeplearning(model_id = 'dl',
                       x = 2:31,
                       y = target,
                       training_frame = train_h2o,
                       validation_frame = validate_h2o,
                       distribution = "multinomial",
                       epochs = 4,
                       activation = 'RectifierWithDropout',
                       hidden_dropout_ratios = c(0, 0),
                       seed = 1234)
# #Set timer:
# timer <- proc.time()
# 
# #Set Grid parameters:
# hidden_opt <- list(c(32,32,32), c(5,25,75), c(100,100,100))
# l1_opt     <- c(1e-5, 1e-4,1e-3)
# hidden_drpoutRatios <- list(c(0.5,0.5,0.5), c(0.5,0.3,0.2), c(0.1,0.2,0.8))
# hyper_pars <- list(hidden = hidden_opt, hidden_dropout_ratios = hidden_drpoutRatios, l1 = l1_opt)
# #Building Grid models:
# model_grid <- h2o.grid(
#   algorithm = "deeplearning",
#   activation = "RectifierWithDropout",
#   hyper_params = hyper_pars,
#   x = 2:31,
#   y = 1,
#   training_frame = train_h2o,
#   input_dropout_ratio = 0.2,
#   balance_classes = T,
#   momentum_stable = 0.99,
#   nesterov_accelerated_gradient = T,
#   epochs = 50,
#   nfolds = 10,
#   variable_importances = T,
#   keep_cross_validation_predictions = T)
# 
# for (model_id in model_grid@model_ids) {
#   auc <- h2o.auc(h2o.getModel(model_id))
#   print(sprintf('CV set auc: %f', auc))
# }

# # Methods for an H2O model
# h2o.residual_analysis_plot()
# h2o.varimp_plot()
# h2o.shap_explain_row_plot()
# h2o.shap_summary_plot()
# h2o.pd_plot()
# h2o.ice_plot()

#shut down H2O instance
#h2o.shutdown()