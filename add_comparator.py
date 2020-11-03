import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from numpy.linalg import norm 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, activations, optimizers, utils
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import concatenate, Flatten, dot
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from random import randint, randrange, choice
from string import ascii_uppercase

Model = keras.Model
Embedding = layers.Embedding
Input = layers.Input
GRU = layers.GRU
Bidirectional = layers.Bidirectional
Dense = layers.Dense
Activation = layers.Activation
Dropout = layers.Dropout
concatenate = layers.concatenate
Lambda = layers.Lambda
Softmax = layers.Softmax
plot_model = utils.plot_model

# Define boundaries for street number integers
# While the lower boundary is self explanatory, the upper bound is an arbitrary choice
LOWER_BOUND = 1
UPPER_BOUND = 500

# Clean wrapper to generate random number
# Optionally pass another number (`no_match`) to assert that the number generated is different,
# i.e. use to ensure two randomly generated numbers are different
def rand(no_match=None):
    if no_match:
        while True:
            i = randint(LOWER_BOUND, UPPER_BOUND)

            if i != no_match:
                return i

    return randint(LOWER_BOUND, UPPER_BOUND)

# Returns a randomly generated range and optionally a number that is either
# within the range (num_within=True), or out of the range (num_within=False).
# If no additional argument is passed, only the range (first, last) is returned
def rand_range(num_within=None):
    start = rand(no_match=UPPER_BOUND)
    # NOTE: This ensures `stop < UPPER_BOUND` which is necessary
    # to avoid any issues when choosing a random number outside the range
    delta = randrange(start, UPPER_BOUND)
    stop = randrange(start + 2, start + delta, 2)

    
    if num_within:
        return (start, stop, randrange(start, stop, step=2))

    elif num_within is False:
        choices = [
      randrange(start + 1, stop, step=2),
    ]

    # NOTE: make sure that no errors can be caused when
    # selecting a random integer
        if stop + 1 < UPPER_BOUND:
            choices.append(randint(stop + 1, UPPER_BOUND))
        if start - 1 > LOWER_BOUND:
            choices.append(randint(LOWER_BOUND, start - 1))

        i = choice(choices)
        return (start, stop, i)

    else:
        return (start, stop)


# Generate up to 3 random characters to use as pre/suffix
def rand_char():
    return choice([
    f'{choice(ascii_uppercase)}',
    f'{choice(ascii_uppercase)}{choice(ascii_uppercase)}',
    f'{choice(ascii_uppercase)}{choice(ascii_uppercase)}{choice(ascii_uppercase)}',
  ])


def generate_no_match():
    target = [0, 0]
    num1 = rand()
    first, last, i = rand_range(num_within=False)

    cases = [
    # Most basic case, two numbers that are distinct
    (f'{num1}', f'{rand(no_match=num1)}', target),
    # Case 12 vs. 14-8
    (f'{i}', f'{first}-{last}', target),
     ]

    return choice(cases)


def generate_primary_match():
    target = [0, 1]
    num = f'{rand()}'

    cases = [
    # Case 12A vs 12
    (f'{num}{rand_char()}', f'{num}', target),
    # Case 12A vs. 12B
    (f'{num}{rand_char()}', f'{num}{rand_char()}', target),
     ]

    return choice(cases)


def generate_exact_match():
    target = [1, 1]
    num1 = rand()
    first, last, i = rand_range(num_within=True)

    cases = [
    # Most basic  case, two numbers thar are the same
    (f'{num1}', f'{num1}', target),
    # Case where street number is part of a range - like 22 vs. 20-24
    (f'{i}', f'{first}-{last}', target),
     ]

    return choice(cases)


def generate_number_components(num_samples):
  # Functions to generate cases from
  # NOTE: each function returns a tuple: (string1, string2, target)
    fns = [
    generate_no_match,
    generate_primary_match,
    generate_exact_match,
     ]

    for _ in range(0, num_samples):
    # pylint: disable=invalid-name
        fn = choice(fns)

        yield fn()

x1 = []
x2 = []
x3 = []
if __name__ == '__main__':
    for elem in generate_number_components(20000):
        #print(elem)
        x1.append(elem[0])
        x2.append(elem[1])
        x3.append(elem[2])
        #print(elem[2])

x4 = []
for i in x3:
    if i == [0, 0]:
        x4.append(0)
    elif i == [0, 1]:
        x4.append(1)
    else: 
        x4.append(2)
        
df = pd.DataFrame({'add1' : x1,
                   'add2' : x2,
                   'target' : x4 } , columns=['add1', 'add2', 'target'])

INPUT_ENCODING = {
    
    " ": 0,
    "0": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
    "A": 11,
    "B": 12,
    "C": 13,
    "D": 14,
    "E": 15,
    "F": 16,
    "G": 17,
    "H": 18,
    "I": 19,
    "J": 20,
    "K": 21,
    "L": 22,
    "M": 23,
    "N": 24,
    "O": 25,
    "P": 26,
    "Q": 27,
    "R": 28,
    "S": 29,
    "T": 30,
    "U": 31,
    "V": 32,
    "W": 33,
    "X": 34,
    "Y": 35,
    "Z": 36,
    "/": 37,
    "-": 38,
    ",": 39,
    "'": 40,
    "\"": 41,
    "(": 42,
    ")": 43,
    "!": 44,
    "@": 45,
    ".": 46}
def encoding_inputs(add):
    s = re.findall(r'[A-Za-z]+|\d+', add)
    d = [int(i) for i in s if i.isdigit()]
    num = d[0]
    if num % 2 == 0:
        f = 0
    else:
        f = 1
    enc_list = [f]
    enc_list1 = [INPUT_ENCODING[i] for i in list(add)]
    return enc_list + enc_list1
df['add1_enc'] = df.add1.apply(encoding_inputs)
df['add2_enc'] = df.add2.apply(encoding_inputs)

df = shuffle(df)
train = df[:14000]
val = df[14000:17000]
test = df[17000:]

x_train = train.iloc[:,-2:].values
y_train = train.loc[:,'target'].values
x1_train = x_train[:,0]
x2_train = x_train[:,1]
y_train = np_utils.to_categorical(y_train)

x_val = val.iloc[:,-2:].values
y_val = val.loc[:,'target'].values
x1_val = x_val[:,0]
x2_val = x_val[:,1]
y_val = np_utils.to_categorical(y_val)

x_test = test.iloc[:,-2:].values
y_test = test.loc[:,'target'].values
x1_test = x_test[:,0]
x2_test = x_test[:,1]
y_test = np_utils.to_categorical(y_test)

max_length = 48
x1train_padded = pad_sequences(x1_train, maxlen=max_length, padding='post')
x2train_padded = pad_sequences(x2_train, maxlen=max_length, padding='post')

x1val_padded = pad_sequences(x1_val, maxlen=max_length, padding='post')
x2val_padded = pad_sequences(x2_val, maxlen=max_length, padding='post')

x1test_padded = pad_sequences(x1_test, maxlen=max_length, padding='post')
x2test_padded = pad_sequences(x2_test, maxlen=max_length, padding='post')

SEQUENCE_LENGTH = max_length
ENCODED_LENGTH = max_length
INPUT_DIM = max_length
OUTPUT_DIM = 4

# first input
i1 = Input(shape=(SEQUENCE_LENGTH,))
#e1= Embedding(INPUT_DIM, OUTPUT_DIM, input_length=ENCODED_LENGTH)(i1)
#r = Bidirectional(GRU(2, return_sequences=True, reset_after=False))(e1)
#x = Dense(3, activation='relu')(r)
#x = Dropout(0.1)(x)
#f1 = Flatten()(e1)
# second input
i2 = Input(shape=(SEQUENCE_LENGTH,))
#e2 = Embedding(INPUT_DIM, OUTPUT_DIM, input_length=ENCODED_LENGTH)(i2)
#r = Bidirectional(GRU(2, return_sequences=True, reset_after=False))(e2)
#x = Dense(3, activation='relu')(r)
#x = Dropout(0.1)(x)
#f2 = Flatten()(e2)
# Concatenate (or subtract / multiply or whatever).
x = concatenate([i1, i2])
#x = tf.keras.layers.Dot(axes=1, normalize=True)([i1, i2]) 
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.2)(x)
o = Dense(3, activation='softmax')(x)
#o = Dropout(0.2)(x)

model = Model(inputs=[i1, i2], outputs=o)
model.compile(
loss='categorical_crossentropy',
metrics=['accuracy'],
optimizer ='adam')
# summarize layers
print(model.summary())
# plot graph
plot_model(model)

batch_size = 32
epochs = 100
history = model.fit(
x=[x1train_padded, x2train_padded],
y=y_train,
epochs=epochs,
validation_data=([x1val_padded,x2val_padded],y_val),
verbose=0,)
#callbacks=[EarlyStopping(monitor='loss', patience=1)]
# evaluate the model
_, train_acc = model.evaluate(x=[x1train_padded, x2train_padded], y=y_train, verbose=0)
_, test_acc = model.evaluate(x=[x1test_padded,x2test_padded], y=y_test, verbose=0)
print('Train Accuracy is: %.3f, and Test Accuracy is: %.3f' % (train_acc, test_acc))

predictions = model.predict([x1test_padded,x2test_padded])
y_hat = np.argmax(predictions, axis=1)
print(classification_report(test['target'], y_hat, digits=3))