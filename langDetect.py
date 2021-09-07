import unicodedata
from tqdm import tqdm
import re
import numpy as np
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer

leaveIn = ["srp\n", "hrv\n", "eng\n", "pol\n", "fin\n", "als\n", "ces\n"]
#Choosing languages
x_test_file = open(r"data\x_test.txt", "r", encoding="utf8")
x_test = x_test_file.readlines()
x_train_file = open(r"data\x_train.txt", "r", encoding="utf8")
x_train = x_train_file.readlines()
y_test_file = open(r"data\y_test.txt", "r", encoding="utf8")
y_test = y_test_file.readlines()
y_train_file = open(r"data\y_train.txt", "r", encoding="utf8")
y_train = y_train_file.readlines()

y_trainSelected = []
x_trainSelected = []

for i in range(len(y_train)):
    if y_train[i] in leaveIn:
        y_trainSelected.append(y_train[i])
        x_trainSelected.append(x_train[i])

# print("x_trainSelected ",len(x_trainSelected),"y_trainSelected ",len(y_trainSelected))

y_testSelected = []
x_testSelected = []

for i in range(len(y_test)):
    if y_test[i] in leaveIn:
        y_testSelected.append(y_test[i])
        x_testSelected.append(x_test[i])

# print("x_testSelected ",len(x_testSelected),"y_testSelected ",len(y_testSelected))

#Bag of words and ngrams
# x_test_file = open(r"data\x_testSelected.txt", "r", encoding="utf8")
# x_test = x_test_file.readlines()
# x_train_file = open(r"data\x_trainSelected.txt", "r", encoding="utf8")
# x_train = x_train_file.readlines()
# y_test_file = open(r"data\y_testSelected.txt", "r", encoding="utf8")
# y_test = y_test_file.readlines()
# y_train_file = open(r"data\y_trainSelected.txt", "r", encoding="utf8")
# y_train = y_train_file.readlines()

# print(len(x_train))
# print(len(y_train))
# print(len(x_test))
# print(len(y_test))

x_dataset = x_trainSelected + x_testSelected
y_dataset = y_trainSelected + y_testSelected

print(len(x_dataset))
print(len(y_dataset))

for i in range(len(x_dataset)):
    x_dataset[i] = x_dataset[i].lower()
    x_dataset[i] = re.sub(r"\W"," ",x_dataset[i])
    x_dataset[i] = re.sub(r"[0-9]"," ",x_dataset[i])
    x_dataset[i] = re.sub(r"\s+"," ",x_dataset[i])

bigramWordVectorizer = CountVectorizer(analyzer="word", ngram_range=(2,2))
X_bigram_raw = bigramWordVectorizer.fit_transform(x_dataset)
bigramWords = bigramWordVectorizer.get_feature_names()

print(len(bigramWords))

bigramCharVectorizer = CountVectorizer(analyzer="char", ngram_range=(3,3))
X_chargram_raw = bigramCharVectorizer.fit_transform(x_dataset)
bigramChars = bigramCharVectorizer.get_feature_names()

print(len(bigramChars))

if False:
    x_dataset_biwords = []
    for x in tqdm(x_dataset):
        word = []
        for y in bigramWords:
            word.append(x.count(y))
        x_dataset_biwords.append(word)

    print(len(x_dataset_biwords))

    x_np_words = np.array([np.array(xi) for xi in x_dataset_biwords])

x_dataset_chars = []
for x in tqdm(x_dataset):
    char = []
    for y in bigramChars:
        char.append(x.count(y))
    x_dataset_chars.append(char)

print(len(x_dataset_chars))

x_np_chars = np.array([np.array(xi) for xi in x_dataset_chars])

encoder = LabelBinarizer()
transformed_label = encoder.fit_transform(leaveIn)
print(transformed_label)

y_np = encoder.transform(y_dataset)