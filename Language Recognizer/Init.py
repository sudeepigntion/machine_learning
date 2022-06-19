import tensorflow as tf
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

stop = stopwords.words('english')

# print(stop)

train = pd.read_csv("reviews.csv")

# Note *

# We shall make use if the Ney York Times user comments (from Kaggle Datasets)

# Once we create the language classifier, we will use other data
# Until then, let's rely on an English natural language source

print(train.head())

# now we first put everything to lower case

train["Summary_lower"] = train["Summary"].str.lower()
train["Summary_no_punctuation"] = train['Summary_lower'].astype(str).str.replace('[^\w\s]','')

# lets check how the text looks like now! well everything is lowercase and no ugly characters

print(train["Summary_no_punctuation"].head())

Tf_train = train

train['Summary_no_punctuation'] = train['Summary_no_punctuation'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

Tf_train['Summary_no_punctuation'] = train['Summary_no_punctuation'].fillna("fillna")

print(Tf_train['Summary_no_punctuation'].head())

# We first assign our current data frame to another to keep track of our work then we read the first sentence and count words that result to be 21.

print(Tf_train['Summary_no_punctuation'][1])

print(Tf_train['Summary_no_punctuation'][1].count(' '))

max_features = 5000 # we set maximum number of words to 5000
maxlen = 100 # and maximum sequence length to 100

tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_features) # Tokenizer step

tok.fit_on_texts(list(Tf_train['Summary_no_punctuation'])) # Fit to cleaned text
Tf_train = tok.texts_to_sequences(list(Tf_train['Summary_no_punctuation'])) # this is how we create sequences

print(type(Tf_train))

print(len(Tf_train[1]))

print(Tf_train[1])

# Lets execute the pad steps
Tf_train = tf.keras.preprocessing.sequence.pad_sequences(Tf_train, maxlen=maxlen)

print(len(Tf_train[1]))

print(Tf_train[1])

print(train['Summary_no_punctuation'][1])

Tf_train = pd.DataFrame(Tf_train)

print(Tf_train.head())