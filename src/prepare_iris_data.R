# This script prepares the iris dataset for a machine learning model.
# It loads the data, one-hot encodes the labels, splits the data into
# training, validation, and test sets, and saves the prepared data to a
# TSV file.

library(keras)
library(tidyverse)

# Load the iris dataset
data(iris)

# Create labels as a one-hot encoded matrix for Keras compatibility
y = to_categorical(as.integer(pull(iris, Species)) - 1)
colnames(y) = c("setosa", "versicolor", "virginica")

# Add the one-hot encoded labels to the iris dataframe
iris = cbind(iris, y)

# Calculate the number of samples for the training, validation, and test sets
train_size <- ceiling(0.4 * nrow(iris))
val_size <- ceiling(0.3 * nrow(iris))
test_size <- nrow(iris) - train_size - val_size

# Set seed for reproducibility
set.seed(42)

# Shuffle the data to ensure randomness
iris = iris[sample(nrow(iris)), ]

# Add a 'set' column to indicate whether a sample belongs to the training,
# validation, or test set
set = c(rep("training", train_size), rep("validation", val_size), rep("test", test_size))
iris = mutate(iris, set = set)

# Write the prepared data to a TSV file
write_tsv(iris, "iris.tsv")