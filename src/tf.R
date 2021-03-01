wd = getwd()
setwd(wd)

library(keras)


train_file_path <- get_file("../data/train80.csv")
test_file_path <- get_file("eval.csv", TEST_DATA_URL)