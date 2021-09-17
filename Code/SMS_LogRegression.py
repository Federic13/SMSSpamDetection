import pandas as pd
import sklearn
# import imblearn

num_words = 8000  # number of words we consider as features 

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


## Apply Word-level One-Hot Encoding
def tokenization(features_training, features_test, label_training, label_test):

    # We need to transform words into integers. For this purpose we use Tokenizer that transforms arbitrary text into 
    # fixed-length arrays, where the length is the size of the dictionary. 
    # More in detail, the array associated with each sms has on the i-th component the counter of the number of times 
    # the i-th word of the dictionary appears in that sms.
    from tensorflow.keras.preprocessing.text import Tokenizer

    # Determine the Dictionary
    myTokenizer = Tokenizer(num_words = num_words) 
    myTokenizer.fit_on_texts(features_training) # Fit the Tokenizer only on the training set
    #print(myTokenizer.word_index)

    X_training = myTokenizer.texts_to_matrix(features_training, mode='count') # Convert the training set into a matrix

    X_test = myTokenizer.texts_to_matrix(features_test, mode='count') # Apply the trained Tokenizer on the test test

    # Convert our label ("ham", "spam") into 0 and 1
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder().fit(label_training)
    y_training = encoder.transform(label_training)
    y_test = encoder.transform(label_test)


    # The dataset is unbalanced: the majority class is "ham" which accounts for more than 85% of the data.
    # We could choose to under-sample the majority class ("ham") in order to have a balanced dataset and not
    # penalise the "spam" label, but it seems that this step doesn't improve performance.

    # from collections import Counter
    # from imblearn.under_sampling import RandomUnderSampler

    # counter = Counter(y_training) # Summarize class distribution
    # undersample = RandomUnderSampler(sampling_strategy='majority')
    # X_training, y_training = undersample.fit_resample(X_training, y_training) # Fit and apply the transformation

    return X_training, X_test, y_training, y_test


## Train and Evaluate Logistic Regression
def TrainAndEvaluate(X_training, y_training, X_test, y_test):
    
    from sklearn.linear_model import LogisticRegression
    
    logReg = LogisticRegression(penalty='none', max_iter=1000).fit(X_training, y_training)
    Y_pred = logReg.predict(X_test) # Return the predicted class labels for X_test
    proba = logReg.predict_proba(X_test) # Return the predicted probability of each class

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Confusion matrix
    cm = confusion_matrix(y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
    disp.plot()

    from sklearn.metrics import classification_report, accuracy_score 
    print('Logistic regression accuracy: ', accuracy_score(y_test, Y_pred))
    print('Logistic Regression classification report: ', classification_report(y_test, Y_pred)) # Classification report




def main():
    features, label = load_dataset()

    # Split our dataset into training and test set
    from sklearn.model_selection import train_test_split
    features_training, features_test, label_training, label_test = train_test_split(features, label,
                                                                                test_size=0.25, random_state=42)

    # Apply Word-level One-Hot Encoding, by training the Tokenizer on training set and then applying it on test set
    X_training, X_test, y_training, y_test = tokenization(features_training, features_test, label_training, label_test)
    
    TrainAndEvaluate(X_training, y_training, X_test, y_test) # Train and Evaluate Logistic Regression


main()