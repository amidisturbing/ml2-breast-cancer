#change working directory to current dir
wd = getwd()
setwd(wd)
#install.packages("h2o")

#Set seed for reproducibility:
set.seed(2)

#start a local h2o cluster:
localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

train <- h2o.importFile("../data/train80.csv")
test <- h2o.importFile("../data/test20.csv")
#test <- read.csv('../data/test20.csv')[c(-1)] # exclude ID

train$diagnosis <- as.factor(train$diagnosis)
levels(test$diagnosis) <- list("0"="B", "1"="M")

test$diagnosis <- as.factor(test$diagnosis)
levels(test$diagnosis) <- list("0"="B", "1"="M")

#train_h2o <- as.h2o(train)
#test_h2o <- as.h2o(test)
train_h2o <- train
test_h2o <- test
#Set timer:
timer <- proc.time()

#Set Grid parameters:
hidden_opt <- list(c(32,32,32), c(5,25,75), c(100,100,100))
l1_opt     <- c(1e-5, 1e-4,1e-3)
hidden_drpoutRatios <- list(c(0.5,0.5,0.5), c(0.5,0.3,0.2), c(0.1,0.2,0.8))
hyper_pars <- list(hidden = hidden_opt, hidden_dropout_ratios = hidden_drpoutRatios, l1 = l1_opt)
#Building Grid models:
model_grid <- h2o.grid(
  algorithm = "deeplearning",
  activation = "RectifierWithDropout",
  hyper_params = hyper_pars,
  x = 3:32,
  y = 2,
  training_frame = train_h2o,
  input_dropout_ratio = 0.2,
  balance_classes = T,
  momentum_stable = 0.99,
  nesterov_accelerated_gradient = T,
  epochs = 50,
  nfolds = 10,
  variable_importances = T,
  keep_cross_validation_predictions = T)

for (model_id in model_grid@model_ids) {
  auc <- h2o.auc(h2o.getModel(model_id))
  print(sprintf('CV set auc: %f', auc))
}
# # Methods for an H2O model
# h2o.residual_analysis_plot()
# h2o.varimp_plot()
# h2o.shap_explain_row_plot()
# h2o.shap_summary_plot()
# h2o.pd_plot()
# h2o.ice_plot()

#shut down H2O instance
h2o.shutdown()