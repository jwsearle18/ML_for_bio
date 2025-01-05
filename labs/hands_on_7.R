library(tidyverse)
library(keras)

iris = read_tsv('iris.tsv')


training_iris = filter(iris, set == 'training') %>%
  filter(Species == 'versicolor' | Species == 'virginica')

training_features = select(training_iris, Sepal.Length, Sepal.Width, Petal.Length, Petal.Width) %>%
  as.matrix()

training_versicolor_vec = pull(training_iris, versicolor)

validation_iris = filter(iris, set == 'validation') %>%
  filter(Species == 'versicolor' | Species == 'virginica')

validation_features = select(validation_iris, Sepal.Length, Sepal.Width, Petal.Length, Petal.Width) %>%
  as.matrix()

validation_versicolor_vec = pull(validation_iris, versicolor)

test_iris = filter(iris, set == 'test') %>%
  filter(Species == 'versicolor' | Species == 'virginica')

test_features = select(test_iris, Sepal.Length, Sepal.Width, Petal.Length, Petal.Width) %>%
  as.matrix()

test_versicolor_vec = pull(test_iris, versicolor)

model = keras_model_sequential() %>%
  layer_dense(units = 16,
              activation = "relu",
              input_shape = 4) %>%
  layer_dense(units = 16,
              activation = "relu") %>%
  layer_dense(units = 1,
              activation = "sigmoid") %>%
  print()

compile(model, optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = "accuracy")

history = model %>% fit(
  x = training_features,
  y = training_versicolor_vec,
  epochs = 200,
  batch_size = 100,
  validation_data = list(validation_features, validation_versicolor_vec)
)

evaluation = evaluate(model,
                      test_features,
                      test_versicolor_vec)
print(evaluation)

predictions = predict(model, test_features)
print(head(predictions, n = 5))

predictions_tibble = tibble(
  label_binary = test_versicolor_vec,
  prediction = as.vector(predictions)
)
print(predictions_tibble)

predictions_tibble = predictions_tibble %>%
  mutate(label_text = ifelse(label_binary == 1, "versicolor", "virginica")) %>%
  print()

ggplot(predictions_tibble, aes(x=label_text, y=prediction)) +
  geom_boxplot() +
  geom_jitter() +
  labs(x = "Species", y = "Versicolor predicted probability") +
  theme_bw()
  

