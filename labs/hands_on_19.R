library(keras)
library(tidyverse)
library(tm)

tensorflow::set_random_seed(0)

data = read_csv('fibro_abstracts.csv')

abstracts = pull(data, abstract)[1:500]

abstracts = VCorpus(VectorSource(abstracts))
abstracts = tm_map(abstracts, content_transformer(tolower))
abstracts = tm_map(abstracts, stripWhitespace)
removeNumbers = content_transformer(function(x) gsub("\\b\\d+\\b", "", x))
abstracts = tm_map(abstracts, removeNumbers)
abstracts = tm_map(abstracts, removePunctuation)

abstracts = sapply(abstracts, as.character)


max_words = 1000

tokenizer = text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(abstracts)

word_index = tokenizer$word_index

# This identifies all unique words and assigns an integer (token) to each.
print(head(word_index, n = 3))
print(head(tokenizer$word_counts, n = 3))
sequences = texts_to_sequences(tokenizer, abstracts)
print(head(sequences, n = 3))

input_sequences = list()
output_sequences = list()

seq_length = 5

for (sentence_seq in sequences) {
  if (length(sentence_seq) < seq_length + 1) {
    next
  }
  
  for (i in 1:(length(sentence_seq) - seq_length)) {
    seq_in = sentence_seq[i:(i + seq_length - 1)]
    seq_out = sentence_seq[i + seq_length]
    
    input_sequences[[length(input_sequences) + 1]] = seq_in
    output_sequences[[length(output_sequences) + 1]] = seq_out
  }
}

head(input_sequences, n = 3)
head(output_sequences, n = 3)
input_sequence_matrix = pad_sequences(input_sequences,
                                      maxlen = seq_length,
                                      padding = 'pre')

head(input_sequence_matrix)
# This performs one-hot encoding for each word

output_sequence_matrix = to_categorical(output_sequences,
                                        num_classes = max_words + 1)

print(output_sequence_matrix[1:6,1:6])

num_units = 128

model = keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = num_units) %>%
  layer_lstm(units = num_units, dropout = 0.3, return_sequences = TRUE) %>%
  layer_lstm(units = num_units, dropout = 0.3, return_sequences = TRUE) %>%
  layer_lstm(units = num_units, dropout = 0.3, return_sequences = FALSE) %>%
  layer_dense(units = max_words + 1, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = "adam",
  metrics = c('accuracy')
)

history = model %>% fit(
  input_sequence_matrix,
  output_sequence_matrix,
  batch_size = 64,
  epochs = 10,
  validation_split = 0.2
)

predict_next_word = function(seed_text) {
  # Encode and pad the input sequence
  encoded_sequence = texts_to_sequences(tokenizer, seed_text)[[1]]
  encoded_sequence = pad_sequences(list(encoded_sequence), maxlen = seq_length, padding = 'pre')
  
  # Predict the next word probabilities
  next_word_prob = model %>%
    predict(encoded_sequence)
  # Select the word with the highest probability (no randomness)
  predicted_word_index = which.max(next_word_prob)
  predicted_word = names(tokenizer$word_index)[predicted_word_index]
  
  return(predicted_word)
}

predict_next_word("communication between doctors could begin")





