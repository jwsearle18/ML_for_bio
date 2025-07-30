# This script trains a neural network to classify iris flowers as either
# versicolor or virginica. It loads the prepared iris dataset, builds a
# sequential model with two hidden layers, compiles and trains the model,
# evaluates its performance on the test set, and visualizes the predictions.

library(tidyverse)
library(keras)

# Load the prepared iris dataset
iris = read_tsv('iris.tsv')

# Filter the data to include only versicolor and virginica species
training_iris = filter(iris, set == 'training') %>%
  filter(Species == 'versicolor' | Species == 'virginica')

# Extract the training features and labels
training_features = select(training_iris, Sepal.Length, Sepal.Width, Petal.Length, Petal.Width) %>%
  as.matrix()
training_versicolor_vec = pull(training_iris, versicolor)

# Filter the validation data
validation_iris = filter(iris, set == 'validation') %>%
  filter(Species == 'versicolor' | Species == 'virginica')

# Extract the validation features and labels
validation_features = select(validation_iris, Sepal.Length, Sepal.Width, Petal.Length, Petal.Width) %>%
  as.matrix()
validation_versicolor_vec = pull(validation_iris, versicolor)

# Filter the test data
test_iris = filter(iris, set == 'test') %>%
  filter(Species == 'versicolor' | Species == 'virginica')

# Extract the test features and labels
test_features = select(test_iris, Sepal.Length, Sepal.Width, Petal.Length, Petal.Width) %>%
  as.matrix()
test_versicolor_vec = pull(test_iris, versicolor)

# Build the neural network model
model = keras_model_sequential() %>%
  layer_dense(units = 16,
              activation = "relu",
              input_shape = 4) %>%
  layer_dense(units = 16,
              activation = "relu") %>%
  layer_dense(units = 1,
              activation = "sigmoid") %>%
  print()

# Compile the model
compile(model, optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = "accuracy")

# Train the model
history = model %>% fit(
  x = training_features,
  y = training_versicolor_vec,
  epochs = 200,
  batch_size = 100,
  validation_data = list(validation_features, validation_versicolor_vec)
)

# Evaluate the model on the test set
evaluation = evaluate(model,
                      test_features,
                      test_versicolor_vec)
print(evaluation)

# Make predictions on the test set
predictions = predict(model, test_features)
print(head(predictions, n = 5))

# Create a tibble with the true labels and predictions
predictions_tibble = tibble(
  label_binary = test_versicolor_vec,
  prediction = as.vector(predictions)
)
print(predictions_tibble)

# Add a column with the predicted labels in text format
predictions_tibble = predictions_tibble %>%
  mutate(label_text = ifelse(label_binary == 1, "versicolor", "virginica")) %>%
  print()

# Visualize the predictions
ggplot(predictions_tibble, aes(x=label_text, y=prediction)) +
  geom_boxplot() +
  geom_jitter() +
  labs(x = "Species", y = "Versicolor predicted probability") +
  theme_bw()
  

