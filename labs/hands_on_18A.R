library(keras)
library(tidymodels)
library(tidyverse)

set.seed(0)
tensorflow::set_random_seed(0)

data("iris")

iris = tibble(iris)

data_split = initial_split(iris, prop = 0.7)

train_data = training(data_split)
test_data = testing(data_split)

train_X = select(train_data, -Species) %>%
  as.matrix() %>%
  scale()

train_center = attr(train_X, "scaled:center")
train_scale = attr(train_X, "scaled:scale")

test_X = select(test_data, -Species) %>%
  as.matrix() %>%
  scale(center=train_center, scale=train_scale)

input_dim = ncol(train_X)

autoencoder = keras_model_sequential() %>%
  layer_dense(units = 32,
              activation = "relu",
              input_shape = input_dim) %>%
  layer_dense(units = 8,
              activation = "relu") %>%
  layer_dense(units = 2,
              activation = layer_activation_leaky_relu(alpha = 0.01),
              name = "bottleneck") %>%
  layer_dense(units = 8,
              activation = "relu") %>%
  layer_dense(units = 32,
              activation = "relu") %>%
  layer_dense(units = input_dim)

compile(autoencoder, optimizer = "adam",
        loss = "mean_squared_error")

history = autoencoder %>% fit(
  x = train_X,
  y = train_X,
  epochs = 50,
  batch_size = 8,
  validation_split = 0.2
)

autoencoder_inputs = autoencoder$input
autoencoder_outputs = get_layer(autoencoder, "bottleneck")$output

autoencoder_bottleneck = keras_model(
  inputs=autoencoder_inputs,
  outputs=autoencoder_outputs
)

compressed_values = predict(autoencoder_bottleneck, train_X)

compressed_tibble = as_tibble(compressed_values) %>%
  rename(Bottleneck_1 = V1, Bottleneck_2 = V2) %>%
  mutate(Species = train_data$Species)

print(head(compressed_tibble))

# ggplot(compressed_tibble, aes(x = Bottleneck_1, y = Bottleneck_2, color = Species)) +
#   geom_point() +
#   theme_bw() +
#   labs(x = "Bottleneck Dimension 1", y = "Bottleneck Dimension 2")

decoded_values <- predict(autoencoder, train_X)
decoded_tibble <- as_tibble(decoded_values)

decoded_tibble <- decoded_tibble %>%
  mutate(Species = train_data$Species)

decoded_test_values <- predict(autoencoder, test_X)
decoded_test_tibble <- as_tibble(decoded_test_values)

decoded_test_tibble <- decoded_test_tibble %>%
  mutate(Species = test_data$Species)


mod = rand_forest(trees = 100,
                  mode = "classification") %>%
  set_engine("ranger")

recp = recipe(Species ~ ., data = decoded_test_tibble)

wf <- workflow() %>%
  add_model(mod) %>%
  add_recipe(recp)

fitted_model = fit(wf, data = decoded_test_tibble)

predictions_discrete = predict(fitted_model, decoded_test_tibble)

predictions = predictions_discrete %>%
  bind_cols(decoded_test_tibble %>% select(Species))

acc = accuracy(predictions, truth = Species, estimate = .pred_class)
print(acc)





