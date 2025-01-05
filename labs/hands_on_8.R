library(tidyverse)
library(keras)

data = read_tsv("iris.tsv")

set.seed(66)

set_vector = c(rep("training", 60), rep("validation", 45), rep("test", 45))


accuracy_values = vector()

for (i in 1:10) {

  shuffled_set_vector = sample(set_vector)
  
  shuffled_iris = data
  
  shuffled_iris$set = shuffled_set_vector
  
  training_data = filter(shuffled_iris, set == "training") %>%
    select(-set)
  
  X_train = select(training_data, starts_with("Sepal"), starts_with("Petal")) %>%
    as.matrix()
  
  y_train = select(training_data, setosa, versicolor, virginica) %>% as.matrix()
  
  validation_data = filter(shuffled_iris, set == "validation") %>%
    select(-set)
  
  X_val = select(validation_data, starts_with("Sepal"), starts_with("Petal")) %>%
    as.matrix()
  
  y_val = select(validation_data, setosa, versicolor, virginica) %>% as.matrix()
  
  test_data = filter(shuffled_iris, set == "test") %>%
    select(-set)
  
  X_test = select(test_data, starts_with("Sepal"), starts_with("Petal")) %>%
    as.matrix()
  
  y_test = select(test_data, setosa, versicolor, virginica) %>% as.matrix()
  
  model = keras_model_sequential() %>%
    layer_dense(units = 64,
                activation = "relu",
                input_shape = 4) %>%
    layer_dense(units = 64,
                activation = "relu") %>%
    layer_dense(units = 3,
                activation = "softmax")
  
  compile(model, optimizer = "rmsprop",
          loss = "categorical_crossentropy",
          metrics = "accuracy")
  
  history = model %>% fit(
    x = X_train,
    y = y_train,
    epochs = 200,
    batch_size = 32,
    validation_data = list(X_val, y_val)
  )
  
  evaluation = evaluate(model,
                        X_test,
                        y_test)
  accuracy_values[i] = evaluation[["accuracy"]]
}

print(accuracy_values)