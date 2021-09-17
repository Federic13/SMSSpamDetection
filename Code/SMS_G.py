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

