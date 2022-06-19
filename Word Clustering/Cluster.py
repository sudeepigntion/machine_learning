import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

stop = stopwords.words('english')

train_df = pd.read_csv("CommentsApril2017.csv").sample(n=50000)

print(train_df.head())

print(np.unique(train_df["typeOfMaterial"]))

classes = len(np.unique(train_df["typeOfMaterial"]))

Y = train_df["typeOfMaterial"]

encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = tf.keras.utils.to_categorical(
    Y,
    num_classes=classes # this time it is the number of topics
)

print(Y)

train_df["sentence_lower"] = train_df["commentBody"].str.lower()

train_df["sentence_no_punctuation"] = train_df["sentence_lower"].str.replace('[^\w\s]','')
train_df["sentence_no_stopwords"] = train_df["sentence_no_punctuation"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train_df["sentence_no_stopwords"] = train_df["sentence_no_stopwords"].fillna("fillna")

max_features=5000
maxlen=100

tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
tok.fit_on_texts(list(train_df["sentence_no_stopwords"]))

print(tok.word_index)

print(len(tok.word_index))

vocab_size = len(tok.word_index) + 1

train_df = tok.texts_to_sequences(list(train_df["sentence_no_stopwords"]))
train_df = tf.keras.preprocessing.sequence.pad_sequences(train_df, maxlen=maxlen)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df, Y,  test_size=0.1, random_state=42)

embedding_dim = 50

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=maxlen
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(classes, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss=['categorical_crossentropy'],
    metrics=['accuracy']
)

print(model.summary())

model.fit(
    X_train,
    y_train,
    epochs=3
)



