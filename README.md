Code based on: https://medium.com/deep-learning-with-keras/char-level-text-generation-with-an-encoder-decoder-model-3712ef6c1d6f

Text generation using a character-level encoder-decoder model.

The training process is started by executing train_model.py.
Various prameters like batch_size, number of epochs, or the maximum sequence length can be altered using variables in the script.

Input_file is used to specify the data used to train the network. Data can be found in the MailFilter repository.
The file_name variable specifies where the model will be saved after training.
The load variable is a boolean that specifies when the program shall not train a model, but rather load an existing one. Instead of training, the program then loads a network from the location specified in file_name

Both after training and loading a network, the program generates text sequences using various sampling methods.
