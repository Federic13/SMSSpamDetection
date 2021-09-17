import pandas as pd

num_words = 8000  # number of words we consider as features
maxlen = 20  # length of each message

# The dataset we use is known as "SMS Spam Collection v.1" (https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)
# and is a public set composed by 5,574 English, real and non-enconded messages, tagged according being 
# legitimate (ham) or spam. (747 SMS spam messages and 4,827 SMS ham messages)


## Load dataset and call "preprocessing_data" function to clean up text messages
def load_dataset():
    label = pd.read_csv('../main/Data/SMSSpamCollect_label.txt', header=None)[0]
    print("Label details: \n", label.value_counts())

    X_data = pd.read_csv('../main/Data/SMSSpamCollection.txt', sep=";", header=None)[0]

    features = preprocessing_data(X_data)
    print("\nLabel shape: ", label.shape, ", Features shape: ", features.shape)

    return features, label


def preprocessing_data(data):
    for i in range(len(data)):
        data[i] = data[i].replace("\t", "")
        data[i] = data[i].replace("\n", "")

        clean_data = ""
        # We choose to replace strings with only numeric characters with the string "NNNNN" in order to reduce 
        # the amount of irrilevant words that will be tokenized
        for word in data[i].split():
            clean_data += " NNNNN" if word.isnumeric() else (" " + word)

        data[i] = clean_data
    return data


## Apply Word-level One-Hot Encoding and truncate each message to have the same fixed length (20 words)
def tokenizationAndPadding(features_training, features_test, label_training, label_test):

    # We need to split each message into individual words and to associate each word to a single integer.
    # We do this via the tool Tokenizer, that allows to associate each word into a dictionary,
    # where each unique word gets a different ID.
    from tensorflow.keras.preprocessing.text import Tokenizer

    myTokenizer = Tokenizer(num_words=num_words) 
    myTokenizer.fit_on_texts(features_training) # Fit the Tokenizer only on the training set
    X_training = myTokenizer.texts_to_sequences(features_training)
    #print(myTokenizer.word_index) # Return a dictionary of words and associated integers

    X_test = myTokenizer.texts_to_sequences(features_test) # Apply the trained Tokenizer on the test test
    
    # Now we want each message to have the same length: 20 words 
    # (if a message is of length greater than 20, then the last 20 words will be considered)
    from keras.preprocessing import sequence

    X_training = sequence.pad_sequences(X_training, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    # Convert our label ("ham", "spam") into 0 and 1
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder().fit(label_training)
    y_training = encoder.transform(label_training)
    y_test = encoder.transform(label_test)

    return X_training, X_test, y_training, y_test


## Train and Evaluate the RNN (this model comes from a previous model selection process)
def TrainAndEvaluate(X_training, X_test, y_training, y_test):

    # We use an Embedding that takes as input arrays of length 20 (this is the number of words in each message) 
    # and returns two-dimensional arrays of shape (20, 6). Then, we have a Bidirectional GRU layer with output space 
    # of dimension 6. Lastly, there is a Dense layer with sigmoid activation function and only one unit.

    from keras.models import Input, Model
    from keras.layers.core import Dense
    from keras.layers import Embedding, Bidirectional, GRU

    embedding_space= 6

    inputs = Input((X_training.shape[-1],)) # shape=(None, 20)
    hidden = Embedding(num_words, embedding_space, input_length=maxlen)(inputs) # shape=(None, 20, 6)
    hidden2 = Bidirectional(GRU(6, dropout=0.1, recurrent_dropout=0.2))(hidden) # shape=(None, 12)
    outputs = Dense(1, activation='sigmoid')(hidden2) # shape=(None, 1)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc']) # Compile the NN for training

    history = model.fit(X_training, y_training, batch_size=100, epochs=40, verbose=0, 
                                                    validation_data= (X_test, y_test)) # Fit the NN for 40 epochs

    #model.save_weights('../SMSSpamDetection/BidirectGRU.h5')

    y_predict = model.predict(X_test) # Return the predicted labels that are real value between 0 and 1
    y_pred = [1 if x > 0.5 else 0 for x in y_predict]
    # We define a threshold (0.5) to decide when the predicted label is 0 or 1

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
    disp.plot()

    from sklearn.metrics import accuracy_score, classification_report
    print('GRU accuracy: ', accuracy_score(y_test, y_pred))
    print('GRU classification report: ', classification_report(y_test, y_pred)) # Classification report

    # Data Visualisation: Accuracy of the training and test set as the epochs change
    import matplotlib.pyplot as plt
    acc= history.history['acc']
    val_acc= history.history['val_acc'] 

    epochs= range(1, len(acc)+1)
    plt.plot(epochs, acc, 'b-', label= 'Training acc')
    plt.plot(epochs, val_acc, 'r+', label= 'Testing acc')
    plt.title('Training and test accuracy as function of the number of epochs')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('Training and test accuracy of RNN.png', dpi=300)
    
    return y_pred


## Identify misclassified messages
def MessagesMisclassified(X_test, y_pred, y_test, features_test):

    # Identify messages for which our network has made an error in the estimated label 
    wrong = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]

    false_positive = [i for i in wrong if y_pred[i] == 1] # Identify false positive
    false_negative = set(wrong) - set(false_positive) # Identify false negative

    print('\nFalse Positive: ')
    [print(features_test.iloc[i], '\n', X_test[i], '\n') for i in false_positive]
    print('\nFalse Negative: ')
    [print(features_test.iloc[i], '\n', X_test[i], '\n') for i in false_negative]




def main(): 

    features, label = load_dataset()
    
    # Split the dataset into Training set and Test set
    from sklearn.model_selection import train_test_split
    features_training, features_test, label_training, label_test = train_test_split(features, label,
                                                                                    test_size= 0.25, random_state= 42)
    
    # Apply Word-level One-Hot Encoding, by training the Tokenizer on training set and then applying it on test set
    X_training, X_test, y_training, y_test = tokenizationAndPadding(features_training, features_test, 
                                                                                    label_training, label_test)
    
    y_pred = TrainAndEvaluate(X_training, X_test, y_training, y_test) # Train and Evaluate RNN
    
    # To investigate further, we can search for all messages for which the wrong class has been estimated
    MessagesMisclassified(X_test, y_pred, y_test, features_test)


main()

