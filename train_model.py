#source: https://colab.research.google.com/drive/1mhabbI7Af1AfLFhen8C69LK2U-YVbW1A?usp=sharing#scrollTo=4t1RHHOc5mI0
import os

import tensorflow as tf
import numpy as np
import re
import string

from tensorflow.keras import layers



print(tf.config.list_physical_devices())
print(tf.config.list_physical_devices('GPU'))
#tf.debugging.set_log_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')


load = 0
file_name = 'model_prediction_encoder_decoder_remove_unk_seqlen_30_test_25_epochs_Adam_lr_001_v3'
input_file = "all_mails_3.csv"
batch_size = 64
maxlen = 30
#raw_data_ds = tf.data.TextLineDataset(["nietzsche.txt"])
raw_data_ds = tf.data.TextLineDataset([input_file])


for elems in raw_data_ds.take(10):
    print(elems.numpy().decode("utf-8"))

text = ""
for elem in raw_data_ds:
    text = text + (elem.numpy().decode('utf-8'))

text = re.sub(r'\ufffd', r'', text, flags=re.MULTILINE)  # remove unknown characters

print(text[:1000])

print("Corpus length:", int(len(text)/1000),"K chars")

chars = sorted(list(set(text)))
print("Total disctinct chars:", len(chars))

# cut the text in semi-redundant sequences of maxlen characters

step = 3
input_chars = []
next_char = []

for i in range(0, len(text) - maxlen, step):
    input_chars.append(text[i : i + maxlen])
    next_char.append(text[i + maxlen])

print("Number of sequences:", len(input_chars))
print("input X  (input_chars)  --->   output y (next_char) ")

for i in range(5):
  print( input_chars[i],"   --->  ", next_char[i])


X_train_ds_raw=tf.data.Dataset.from_tensor_slices(input_chars)
y_train_ds_raw=tf.data.Dataset.from_tensor_slices(next_char)

for elem1, elem2 in zip(X_train_ds_raw.take(5),y_train_ds_raw.take(5)):
   print(elem1.numpy().decode('utf-8'),"----->", elem2.numpy().decode('utf-8'))

def custom_standardization(input_data):
    lowercase     = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    stripped_num  = tf.strings.regex_replace(stripped_html, "[\d-]", " ")
    stripped_punc  =tf.strings.regex_replace(stripped_num,
                             "[%s]" % re.escape(string.punctuation), "")
    return stripped_punc

def char_split(input_data):
  return tf.strings.unicode_split(input_data, 'UTF-8')

def word_split(input_data):
  return tf.strings.split(input_data)

# Model constants.
max_features = 96           # Number of distinct chars / words
embedding_dim = 16             # Embedding layer output dimension
sequence_length = maxlen       # Input sequence size

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    split=char_split, # word_split or char_split
    output_mode="int",
    output_sequence_length=sequence_length,
)

vectorize_layer.adapt(X_train_ds_raw.batch(batch_size))

print("The size of the vocabulary (number of distinct characters): ", len(vectorize_layer.get_vocabulary()))

print("The first 10 entries: ", vectorize_layer.get_vocabulary()[:10])

vectorize_layer.get_vocabulary()[3]

def vectorize_text(text):
  text = tf.expand_dims(text, -1)
  return tf.squeeze(vectorize_layer(text))

vectorize_text("I am Eve.")

# Vectorize the data.
X_train_ds = X_train_ds_raw.map(vectorize_text)
y_train_ds = y_train_ds_raw.map(vectorize_text)

X_train_ds.element_spec, y_train_ds.element_spec

y_train_ds=y_train_ds.map(lambda x: x[0])

for elem in y_train_ds.take(1):
  print("shape: ", elem.shape, "\n next_char: ",elem.numpy())

X_train_ds.take(1), y_train_ds.take(1)

for (X,y) in zip(X_train_ds.take(5), y_train_ds.take(5)):
  print(X.numpy()," --> ",y.numpy())

train_ds =  tf.data.Dataset.zip((X_train_ds,y_train_ds))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(buffer_size=512).batch(batch_size, drop_remainder=True).cache().prefetch(buffer_size=AUTOTUNE)

print("The size of the dataset (in batches)): ", train_ds.cardinality().numpy())

for sample in train_ds.take(1):
  print("input (X) dimension: ", sample[0].numpy().shape, "\noutput (y) dimension: ",sample[1].numpy().shape)

#--------------------------------sampling-----------------------------------

def softmax(z):
   return np.exp(z)/sum(np.exp(z))

def greedy_search(conditional_probability):
  return (np.argmax(conditional_probability))

def temperature_sampling (conditional_probability, temperature=1.0):
  conditional_probability = np.asarray(conditional_probability).astype("float64")
  conditional_probability = np.log(conditional_probability) / temperature
  reweighted_conditional_probability = softmax(conditional_probability)
  probas = np.random.multinomial(1, reweighted_conditional_probability, 1)
  return np.argmax(probas)

def top_k_sampling(conditional_probability, k):
  top_k_probabilities, top_k_indices= tf.math.top_k(conditional_probability, k=k, sorted=True)
  top_k_probabilities= np.asarray(top_k_probabilities).astype("float32")
  top_k_probabilities= np.squeeze(top_k_probabilities)
  top_k_indices = np.asarray(top_k_indices).astype("int32")
  top_k_redistributed_probability=softmax(top_k_probabilities)
  top_k_redistributed_probability = np.asarray(top_k_redistributed_probability).astype("float32")
  sampled_token = np.random.choice(np.squeeze(top_k_indices), p=top_k_redistributed_probability)
  return sampled_token

#--------------------------------Attention Layer-----------------------------------

LSTMoutputDimension = 64


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, verbose=0):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.verbose = verbose

    def call(self, query, values):
        if self.verbose:
            print('\n******* Bahdanau Attention STARTS******')
            print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
            print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', values.shape)

        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        if self.verbose:
            print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
        if self.verbose:
            print('score: (batch_size, max_length, 1) ', score.shape)
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        if self.verbose:
            print('attention_weights: (batch_size, max_length, 1) ', attention_weights.shape)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        if self.verbose:
            print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ', context_vector.shape)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        if self.verbose:
            print('context_vector after reduce_sum: (batch_size, hidden_size) ', context_vector.shape)
            print('\n******* Bahdanau Attention ENDS******')
        return context_vector, attention_weights

#--------------------------------Encoder Decoder-----------------------------------

verbose = 0
# See all debug messages

# batch_size=1
if verbose:
    print('***** Model Hyper Parameters *******')
    print('latentSpaceDimension: ', LSTMoutputDimension)
    print('batch_size: ', batch_size)
    print('sequence length (n_timesteps_in): ', max_features)
    print('n_features: ', embedding_dim)

    print('\n***** TENSOR DIMENSIONS *******')

# The first part is encoder
# A integer input for vocab indices.
encoder_inputs = tf.keras.Input(shape=(sequence_length,), dtype="int64", name='encoder_inputs')
# encoder_inputs = Input(shape=(n_timesteps_in, n_features), name='encoder_inputs')

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
embedding = layers.Embedding(max_features, embedding_dim)
embedded = embedding(encoder_inputs)

encoder_lstm = layers.LSTM(LSTMoutputDimension, return_sequences=True, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(embedded)

if verbose:
    print('Encoder output shape: (batch size, sequence length, latentSpaceDimension) {}'.format(encoder_outputs.shape))
    print('Encoder Hidden state shape: (batch size, latentSpaceDimension) {}'.format(encoder_state_h.shape))
    print('Encoder Cell state shape: (batch size, latentSpaceDimension) {}'.format(encoder_state_c.shape))
# initial context vector is the states of the encoder
encoder_states = [encoder_state_h, encoder_state_c]
if verbose:
    print(encoder_states)
# Set up the attention layer
attention = BahdanauAttention(LSTMoutputDimension, verbose=verbose)

# Set up the decoder layers
decoder_inputs = layers.Input(shape=(1, (embedding_dim + LSTMoutputDimension)), name='decoder_inputs')
decoder_lstm = layers.LSTM(LSTMoutputDimension, return_state=True, name='decoder_lstm')
decoder_dense = layers.Dense(max_features, activation='softmax', name='decoder_dense')

all_outputs = []

# 1 initial decoder's input data
# Prepare initial decoder input data that just contains the start character
# Note that we made it a constant one-hot-encoded in the model
# that is, [1 0 0 0 0 0 0 0 0 0] is the first input for each loop
# one-hot encoded zero(0) is the start symbol
inputs = np.zeros((batch_size, 1, max_features))
inputs[:, 0, 0] = 1
# 2 initial decoder's state
# encoder's last hidden state + last cell state
decoder_outputs = encoder_state_h
states = encoder_states
if verbose:
    print('initial decoder inputs: ', inputs.shape)

# decoder will only process one time step at a time.
for _ in range(1):

    # 3 pay attention
    # create the context vector by applying attention to
    # decoder_outputs (last hidden state) + encoder_outputs (all hidden states)
    context_vector, attention_weights = attention(decoder_outputs, encoder_outputs)
    if verbose:
        print("Attention context_vector: (batch size, units) {}".format(context_vector.shape))
        print("Attention weights : (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
        print('decoder_outputs: (batch_size,  latentSpaceDimension) ', decoder_outputs.shape)

    context_vector = tf.expand_dims(context_vector, 1)
    if verbose:
        print('Reshaped context_vector: ', context_vector.shape)

    # 4. concatenate the input + context vectore to find the next decoder's input
    inputs = tf.concat([context_vector, tf.dtypes.cast(inputs, tf.float32)], axis=-1)

    if verbose:
        print('After concat inputs: (batch_size, 1, n_features + hidden_size): ', inputs.shape)

    # 5. passing the concatenated vector to the LSTM
    # Run the decoder on one timestep with attended input and previous states
    decoder_outputs, state_h, state_c = decoder_lstm(inputs,
                                                     initial_state=states)
    # decoder_outputs = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))

    outputs = decoder_dense(decoder_outputs)
    # 6. Use the last hidden state for prediction the output
    # save the current prediction
    # we will concatenate all predictions later
    outputs = tf.expand_dims(outputs, 1)
    all_outputs.append(outputs)
    # 7. Reinject the output (prediction) as inputs for the next loop iteration
    # as well as update the states
    inputs = outputs
    states = [state_h, state_c]

# 8. After running Decoder for max time steps
# we had created a predition list for the output sequence
# convert the list to output array by Concatenating all predictions
# such as [batch_size, timesteps, features]
decoder_outputs = layers.Lambda(lambda x: layers.concatenate(x, axis=1))(all_outputs)

# 9. Define and compile model
model_encoder_decoder_Bahdanau_Attention = tf.keras.Model(encoder_inputs,
                                                 decoder_outputs, name='model_encoder_decoder')

#--------------------------------Compile-----------------------------------

#model_encoder_decoder_Bahdanau_Attention.trainable = True

model_encoder_decoder_Bahdanau_Attention.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                                                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_encoder_decoder_Bahdanau_Attention.summary()

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints/' + file_name
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

#--------------------------------Train or load-----------------------------------
if not load:
    model_encoder_decoder_Bahdanau_Attention.fit(train_ds, epochs=25, callbacks=[checkpoint_callback])

    model_encoder_decoder_Bahdanau_Attention.save(file_name)
    os.makedirs(f'saved_layers/{file_name}/input', True)
    for layer in model_encoder_decoder_Bahdanau_Attention.layers:
        weights = layer.get_weights()
        if weights != []:
            np.savez(f'saved_layers/{file_name}/{layer.name}.npz', weights)
    np.savez(f'saved_layers/{file_name}/input/input_file.npz', input_file)
else:
    #load layer weights
    print('Loading layer weights from files')
    w_bahdanau_attention = np.load(f'saved_layers/{file_name}/bahdanau_attention.npz', allow_pickle=True)
    w_decoder_dense = np.load(f'saved_layers/{file_name}/decoder_dense.npz', allow_pickle=True)
    w_encoder_lstm = np.load(f'saved_layers/{file_name}/encoder_lstm.npz', allow_pickle=True)
    w_decoder_lstm = np.load(f'saved_layers/{file_name}/decoder_lstm.npz', allow_pickle=True)
    w_embedding = np.load(f'saved_layers/{file_name}/embedding.npz', allow_pickle=True)
    #set layer weights
    model_encoder_decoder_Bahdanau_Attention.layers[1].set_weights(w_embedding['arr_0'])
    model_encoder_decoder_Bahdanau_Attention.layers[2].set_weights(w_encoder_lstm['arr_0'])
    model_encoder_decoder_Bahdanau_Attention.layers[3].set_weights(w_bahdanau_attention['arr_0'])
    model_encoder_decoder_Bahdanau_Attention.layers[6].set_weights(w_decoder_lstm['arr_0'])
    model_encoder_decoder_Bahdanau_Attention.layers[7].set_weights(w_decoder_dense['arr_0'])
    print('Succesfully loaded layer weights from files')
#--------------------------------Inference Model-----------------------------------

# The first part is encoder
# A integer input for vocab indices.
encoder_inputs = tf.keras.Input(shape=(sequence_length,), dtype="int64", name='encoder_inputs')

embedded= embedding(encoder_inputs)
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(embedded)

encoder_states = [encoder_state_h, encoder_state_c]

all_outputs = []

inputs = np.zeros((1, 1, max_features))
inputs[:, 0, 0] = 1

decoder_outputs = encoder_state_h
states = encoder_states

context_vector, attention_weights=attention(decoder_outputs, encoder_outputs)
context_vector = tf.expand_dims(context_vector, 1)
inputs = tf.concat([context_vector, tf.dtypes.cast(inputs, tf.float32)], axis=-1)
decoder_outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
outputs = decoder_dense(decoder_outputs)
outputs = tf.expand_dims(outputs, 1)


# 9. Define and compile model
model_encoder_decoder_Bahdanau_Attention_PREDICTION = tf.keras.Model(encoder_inputs,
                                                 outputs, name='model_encoder_decoder')

#--------------------------------Token decoding-----------------------------------

def decode_sequence (encoded_sequence):
  decoded_sequence=[]
  for token in encoded_sequence:
    decoded_sequence.append(vectorize_layer.get_vocabulary()[token])
  sequence= ''.join(decoded_sequence)
  print("\t",sequence)
  return sequence

#--------------------------------Text Generation-----------------------------------

def generate_text(model, seed_original, step):
    seed = vectorize_text(seed_original)
    print("The prompt is")
    decode_sequence(seed.numpy().squeeze())
    greedy_sequences = []
    temp_sequences = []
    topk_sequences = []

    seed = vectorize_text(seed_original).numpy().reshape(1, -1)
    # Text Generated by Greedy Search Sampling
    generated_greedy_search = (seed)
    for i in range(step):
        predictions = model.predict(seed)
        next_index = greedy_search(predictions.squeeze())
        generated_greedy_search = np.append(generated_greedy_search, next_index)
        seed = generated_greedy_search[-sequence_length:].reshape(1, sequence_length)
    print("Text Generated by Greedy Search Sampling:")
    greedy_sequences.append(decode_sequence(generated_greedy_search))

    # Text Generated by Temperature Sampling
    print("Text Generated by Temperature Sampling:")
    for temperature in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.2]:
        print("\ttemperature: ", temperature)
        seed = vectorize_text(seed_original).numpy().reshape(1, -1)
        generated_temperature = (seed)
        for i in range(step):
            predictions = model.predict(seed)
            next_index = temperature_sampling(predictions.squeeze(), temperature)
            generated_temperature = np.append(generated_temperature, next_index)
            seed = generated_temperature[-sequence_length:].reshape(1, sequence_length)
        temp_sequences.append(decode_sequence(generated_temperature))

    # Text Generated by Top-K Sampling
    print("Text Generated by Top-K Sampling:")
    for k in [2, 3, 4, 5]:
        print("\tTop-k: ", k)
        seed = vectorize_text(seed_original).numpy().reshape(1, -1)
        generated_top_k = (seed)
        for i in range(step):
            predictions = model.predict(seed)
            next_index = top_k_sampling(predictions.squeeze(), k)
            generated_top_k = np.append(generated_top_k, next_index)
            seed = generated_top_k[-sequence_length:].reshape(1, sequence_length)
        topk_sequences.append(decode_sequence(generated_top_k))

    print('Greedy:')
    for seq in greedy_sequences:
        print(seq)
        print('---------------------')

    print('Temp:')
    for seq in temp_sequences:
        print(seq)
        print('---------------------')

    print('Top-K:')
    for seq in topk_sequences:
        print(seq)
        print('---------------------')

#--------------------------------Test the model-----------------------------------
if not load:
    generate_text(model_encoder_decoder_Bahdanau_Attention_PREDICTION,
                  "Dear customer",
                  100)
else:
    #generate_text(model_encoder_decoder_Bahdanau_Attention_PREDICTION,
     #             "Dear student, we regret to inform you",
      #            300)
    generate_text(model_encoder_decoder_Bahdanau_Attention_PREDICTION,
                  "Notification: We regret to inform you that your account will be suspended",
                  300)
    #generate_text(model_encoder_decoder_Bahdanau_Attention_PREDICTION,
     #             "Warning",
      #            300)
print('Done')