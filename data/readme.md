# original data

data.csv is the original data from kaggle: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

the basic description is "Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. "

we changed the diagnosis from B, M to 0,1 (put differently: B->0, M->1)

# split

The data in data.csv was split into test and train set with a stratified 80 / 20 split.

we do our training for both / all methods on the training set, we individually split the training set further into train and validation sets. This split may be different for the different methods we apply.
And we may use several train / val splits.

Only when comparing methods we will use the test set.
