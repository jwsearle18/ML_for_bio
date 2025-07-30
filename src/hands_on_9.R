library(tidyverse)
library(keras)

set.seed(82)

# Function to preprocess data
preprocess_data <- function(data) {
  shuffled <- slice_sample(data, prop=1)
  num_total <- nrow(data)
  num_training <- ceiling(num_total * 0.6)
  num_validation <- ceiling(num_total * 0.2)
  num_test <- num_total - (num_training + num_validation)
  
  training_data <- shuffled[1:num_training,]
  validation_data <- shuffled[(num_training + 1):(num_training + num_validation),]
  test_data <- shuffled[(num_total - num_test + 1):num_total,]
  
  training_features <- scale(as.matrix(select(training_data, -Class)))
  training_labels <- as.numeric(as.factor(training_data$Class)) - 1
  validation_features <- scale(as.matrix(select(validation_data, -Class)))
  validation_labels <- as.numeric(as.factor(validation_data$Class)) - 1
  test_features <- scale(as.matrix(select(test_data, -Class)))
  test_labels <- as.numeric(as.factor(test_data$Class)) - 1
  
  list(
    training = list(features = training_features, labels = training_labels),
    validation = list(features = validation_features, labels = validation_labels),
    test = list(features = test_features, labels = test_labels)
  )
}

# Function to create and train model
train_model <- function(training_data, validation_data, use_class_weights = FALSE) {
  input_dim <- ncol(training_data$features)
  
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = input_dim) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    optimizer = optimizer_rmsprop(learning_rate = 0.001),
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )
  
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE)
  
  if (use_class_weights) {
    class_weights <- list("0" = 1, "1" = sum(training_data$labels == 0) / sum(training_data$labels == 1))
  } else {
    class_weights <- NULL
  }
  
  history <- model %>% fit(
    x = training_data$features,
    y = training_data$labels,
    epochs = 100,
    batch_size = 32,
    validation_data = list(validation_data$features, validation_data$labels),
    callbacks = early_stop,
    class_weight = class_weights
  )
  
  model
}

# Function to evaluate model
evaluate_model <- function(model, test_data) {
  evaluation <- evaluate(model, test_data$features, test_data$labels)
  evaluation[2]  # Return the accuracy (assuming it's the second element)
}

# Main function to run experiment
run_experiment <- function(data, num_iterations = 3, use_class_weights = FALSE) {
  accuracies <- numeric(num_iterations)
  
  for (i in 1:num_iterations) {
    preprocessed_data <- preprocess_data(data)
    model <- train_model(preprocessed_data$training, preprocessed_data$validation, use_class_weights)
    accuracies[i] <- evaluate_model(model, preprocessed_data$test)
  }
  
  mean(accuracies)
}

# Load data
data <- read_csv('diabetes.csv') %>% rename(Class = Outcome)

# Run experiments
baseline_accuracy <- run_experiment(data, num_iterations = 3)
weighted_accuracy <- run_experiment(data, num_iterations = 3, use_class_weights = TRUE)

# Main function to run experiment
run_experiment <- function(data, num_iterations = 3, use_class_weights = FALSE) {
  accuracies <- numeric(num_iterations)
  
  for (i in 1:num_iterations) {
    preprocessed_data <- preprocess_data(data)
    model <- train_model(preprocessed_data$training, preprocessed_data$validation, use_class_weights)
    accuracies[i] <- evaluate_model(model, preprocessed_data$test)
    cat("Iteration", i, "Accuracy:", accuracies[i], "\n")
  }
  
  mean_accuracy <- mean(accuracies)
  cat("Average Accuracy:", mean_accuracy, "\n\n")
  
  mean_accuracy
}

# Load data
data <- read_csv('diabetes.csv') %>% rename(Class = Outcome)

# Run experiments
cat("Baseline Model:\n")
baseline_accuracy <- run_experiment(data, num_iterations = 3)

cat("Model with Class Weights:\n")
weighted_accuracy <- run_experiment(data, num_iterations = 3, use_class_weights = TRUE)

# Print overall results
cat("Overall Results:\n")
cat("Average Accuracy (Baseline):", baseline_accuracy, "\n")
cat("Average Accuracy (With Class Weights):", weighted_accuracy, "\n")