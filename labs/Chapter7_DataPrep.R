library(keras)
library(tidyverse)

# Load the iris dataset
data(iris)

# Create labels as a one-hot encoded matrix for Keras compatibility
y = to_categorical(as.integer(pull(iris, Species)) - 1)
colnames(y) = c("setosa", "versicolor", "virginica")

# Add the one-hot encoded labels
iris = cbind(iris, y)

# Calculate the number of samples for the training, validation, and test sets
train_size <- ceiling(0.4 * nrow(iris))
val_size <- ceiling(0.3 * nrow(iris))
test_size <- nrow(iris) - train_size - val_size

# Set seed for reproducibility
set.seed(42)

# Shuffle the data to ensure randomness
iris = iris[sample(nrow(iris)), ]

# Add the indicator column for training, validation, and test sets
set = c(rep("training", train_size), rep("validation", val_size), rep("test", test_size))
iris = mutate(iris, set = set)

write_tsv(iris, "iris.tsv")