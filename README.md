# SMS Spam Detection

The dataset we use is known as "SMS Spam Collection v. 1" and is a [public set](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/) of 5574 English, real and non-enconded messages, tagged according being legitimate (ham) or spam (4827 SMS ham messages and 747 SMS spam messages).

We use two different approaches: a machine learning and a deep learning algorithm. The first is Logistic Regression, which is perhaps one of the best known ML algorithms. The second is a Recurrent Neural Network, which was created with the intention of analyzing temporal dependencies, but is widely used for speech recognition. We use two slightly different preprocessing techniques and then analyze the results.

## Data preprocessing
First of all, for both approaches we clean messages from special characters ("\t", "\n") and then we replace every string with only numeric characters with the string "NNNNN". This idea is similar to what is described in ["Towards SMS Spam Filtering: Results under a New Dataset"](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/IJISS13.pdf) (Almeida, Hidalgo, Silva) and is used to reduce the amount of words we pass to Tokenizer for encoding. 

For Logistic Regression we use a "texts_to_matrix" Tokenizer. This method transforms an array of arbitrary length into a fixed-length array, where the length is the number of words in the dictionary, in our case 8000. More in detail, the array returned for a given message has on the i-th component the frequency in this message of the word associated with the integer i by the tokenizer. It is worth noting that this method does not take into account the order of the words in the message, but considers all the words contained in a text message.

Instead, for Recurrent Neural network we use a "texts_to_sequences" Tokenizer. It maps each word in the dictionary to an integer and then simply replaces the text message with an array of the same length, where on the i-th position is the integer associated with the i-th word in the text message. In this way, we're considering the order in which the words are written in the message. Then, only for this tokenization method, we pad or truncate each array so that they are all the same length, in our case 20 elements. We note that if an array is longer than 20 elements, then the last 20 integers will be retained, i.e. the last 20 words in the text message.

## Models
Our first model is Logistic Regression with a very large number of features: 8000. However, the input arrays are sparse (most components are 0) and this is probably why this algorithm works better without coefficient regularization. 

The second model is composed by an Embedding layer followed by a Bidirectional GRU.
The Embedding layer maps each word, represented by an integer, into a dense-array of fixed length, this length is known as "embedding_space". At theoretical level, the Embedding maps words into a meaning-space, so that words with similar meanings are represented by similar arrays (or arrays that have small distance). Our Embedding takes as input arrays of 20 elements and returns two-dimensional arrays of size (20,6), so the "embedding_space" is 6.
Next, there is a Bidirectional GRU layer with output space of dimension 6. It is in GRU layer that words are seen together and their dependencies are modeled; besides, because of bidirectionality, words take on meaning based on context by considering both previous and following words. 
Lastly, we have a Dense layer with a single unit and a Sigmoid activation function that aggregates information and returns the output. 

## Results
a) Logistic Regression

Accuracy: 0.9870875179340028

Confusion Matrix: 
|                    | Predicted label: Ham (0)  |  Predicted label: Spam (1)   |
|          :---      |          :---:            |           :---:              |
|True label: Ham (0) |  1198                     |                            5 |
|True label: Spam (1)|    13                     |                           178|

Classification Report: 

|  Class   |  Precision  |  Recall    | F1-score   |   Support |
|   :---   |    :---:    |    :---:   |   :---:    |    :---:  |
|  Ham (0) |    0.99     |    1.00    |  0.99      |    1203   |
| Spam (1) |    0.97     |   0.93     |   0.95     |    191    |

b) Recurrent Neural Network

Accuracy: 0.9913916786226685

Confusion Matrix: 
|                    | Predicted label: Ham (0)  |  Predicted label: Spam (1)   |
|          :---      |          :---:            |           :---:              |
|True label: Ham (0) |  1201                     |                             2|
|True label: Spam (1)|    10                     |                           181|

Classification Report: 

|  Class   |  Precision  |  Recall    | F1-score   |   Support |
|   :---   |    :---:    |    :---:   |   :---:    |    :---:  |
|  Ham (0) |    0.99     |    1.00    |  1.00      |    1203   |
| Spam (1) |    0.99     |   0.95     |   0.97     |    191    |

These performance results are obtained by training the RNN for 40 epochs. Observing how accuracy on test set changes as a function of the number of epochs, we can say we're not in an overfitting situation, but it is sufficient even only 20 epochs.

<img src="https://user-images.githubusercontent.com/89379052/133884115-b4fb2f9f-ba82-48a2-a64d-412a58c2e6fb.png" width="600">


If we want to investigate further the RNN, we can look for misclassified messages. Here some examples of misclassified: 

1) False Positive: 

-"This is ur face test ( NNNNN NNNNN NNNNN NNNNN NNNNN NNNNN NNNNN NNNNN NNNNN &lt,#&gt, ) select any number i will tell ur face astrology.... am waiting. quick reply..." 

-"on a Tuesday night r u NNNNN real"

2) False Negative:
 
-"Missed call alert. These numbers called but left no message. NNNNN"

-"Sorry I missed your call let's talk when you have the time. I'm on NNNNN"

-"Email AlertFrom: Jeri StewartSize: 2KBSubject: Low-cost prescripiton drvgsTo listen to email call NNNNN"

-"Do you realize that in about NNNNN years, we'll have thousands of old ladies running around with tattoos?"

The first false positive is certainly an ambiguous message; finding strange messages among false positives occurs often, sometimes it is only the sender of the message (whether known or not) that clarifies if the message is really spam. The real problem of our classifier are false negatives; however, we could improve our network by increasing the subset of spam in the dataset, or by applying other techniques that we describe below.

## Possible future improvements and conclusions
As expected, Recurrent Neural Network is more accurate than Logistic Regression. As written before, we could enhance both algorithms by increasing the subset of spam in the dataset. Also, it is possible to improve the preprocessing, we could remove special character ("&lt", "#&gt", ...) and we could use Stemming or Lemmatization to reduce inflection in words. Indeed, preprocessing is essential in NLP and Tokenization is a key step for both our classifiers (for example, if we apply "texts_to_sequences" Tokenizer for Logistic Regression data, we get a Recall of 0.2 for Spam class).
Again, it might be possible to improve the structure of the RNN: the proposed RNN is obtained with a model selection on the recurrent layer (SimpleRNN, LSTM or GRU) and its direction (whether Bidirectional or not) and on the "embedding_space" dimensionality by fixing the size of the input arrays. It might be possibile to find a better classifier by changing the size of input arrays from 20. Adding another recurrent layer (SimpleRNN, LST or GRU) after GRU could be another idea to enhance the network, in order to increase the  network complexity and thus its learning capacity.


