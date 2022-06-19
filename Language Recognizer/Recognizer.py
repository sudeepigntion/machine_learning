import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv("languages.csv", sep='\s*,\s*',header=0, engine='python') # here we have the dataset we extracted

print(train_df.columns.tolist())

print(train_df.head())

print(len(train_df))

# A key step is to label encode the target variable from text to number

Y = train_df["Language"]

encoder = LabelEncoder()

encoder.fit(Y)

Y = encoder.transform(Y)

print(Y)

Y = tf.keras.utils.to_categorical(
    Y,
    num_classes=6 # equals to the number of languages
)

print(Y)

print(train_df)

# As we mentioned, we will perform the previous text processing steps except for stopwords removal

train_df["sentence_lower"] = train_df["Sentence"]
train_df["sentence_no_punctuation"] = train_df["sentence_lower"].astype(str).str.replace('[^\w\s]','')
train_df["sentence_no_punctuation"] = train_df["sentence_no_punctuation"].fillna("fillna")

max_features=5000 # we set maximum number of words to 5000
maxlen=400 # we set maximum sequence length to 400

tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)

tok.fit_on_texts(list(train_df["sentence_no_punctuation"])) # fit to cleaned text

print(len(tok.word_index))

vocab_size = len(tok.word_index) + 1

print(vocab_size)

train_df = tok.texts_to_sequences(list(train_df["sentence_no_punctuation"]))
train_df = tf.keras.preprocessing.sequence.pad_sequences(train_df, maxlen=maxlen)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df, Y, test_size=0.1, random_state=42)

embedding_dim = 50

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=maxlen
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metric=['accuracy']
)

print(model.summary())

model.fit(
    np.array(X_train),
    np.array(y_train),
    epochs=3,
    verbose=1
)

print(X_train.shape)

model.evaluate(np.array(X_test), np.array(y_test))


print('Afrikaans', encoder.transform(['Afrikaans']))
print('Portuguese', encoder.transform(['Portuguese']))
print('Russian', encoder.transform(['Russian']))
print('Italian', encoder.transform(['Italian']))
print('Spanish', encoder.transform(['Spanish']))

# Predict the sentence

new_text = ["tensorflow is a great tool you can find a lot of tutorials from packt"]

test_test = tok.texts_to_sequences(new_text)

print(test_test)

test_test = tf.keras.preprocessing.sequence.pad_sequences(test_test, maxlen=maxlen)

print(test_test)

np.set_printoptions(suppress=True)

predictions = model.predict(test_test)

print(predictions.argmax())

classes = [
    "Afrikaans",
    "Portuguese",
    "Russian",
    "Italian",
    "Spanish",
    "English"
]

print(predictions)

print(classes[predictions.argmax()])