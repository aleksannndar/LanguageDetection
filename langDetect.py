import unicodedata
from scipy.sparse.construct import rand
from tqdm import tqdm
import re
import numpy as np
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.sparse import csc_matrix, csr_matrix
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.naive_bayes import MultinomialNB

leaveIn = ["srp\n", "hrv\n", "eng\n", "swe\n","bos\n", "pol\n", "nno\n"]
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

y_num = []
for y in y_dataset:
    y_num.append(leaveIn.index(y))

y_np = np.array(y_num)

if False:
    bagOfWordsVectorizer = CountVectorizer(analyzer="word", ngram_range=(1,1))
    X_bow_raw = bagOfWordsVectorizer.fit_transform(x_dataset)
    bagOfWords = bagOfWordsVectorizer.get_feature_names()

    print(len(bagOfWords))
    x_dataset_bow = []
    for x in tqdm(x_dataset):
        bow = []
        for y in bagOfWords:
            bow.append(x.count(y))
        x_dataset_bow.append(bow)

    print(len(x_dataset_bow))

    x_np_bow = np.array([np.array(xi) for xi in x_dataset_bow])

    X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(x_np_bow,y_np,test_size=0.2,random_state=1)
    # X_train_bow, X_val_bow, y_train_bow, y_val_bow = train_test_split(X_train_bow, y_train_bow, test_size=0.5,random_state=1)
    X_train_bow_sparse = csr_matrix(X_train_bow)
    X_test_bow_sparse = csr_matrix(X_test_bow)

    svmClassifier = LinearSVC()
    svmClassifier.fit(X_train_bow_sparse,y_train_bow)
    y_pred_bow = svmClassifier.predict(X_test_bow_sparse)
    print("SMO + Bag of Words tacnost: ",skm.accuracy_score(y_test_bow,y_pred_bow))
    print("SMO + Bag of Words f1 score: ",skm.f1_score(y_test_bow,y_pred_bow,average="micro"))
    print("SMO + Bag of Words precision: ",skm.precision_score(y_test_bow,y_pred_bow,average="micro"))
    print("SMO + Bag of Words recall: ",skm.recall_score(y_test_bow,y_pred_bow,average="micro"))
    dispBow = skm.plot_confusion_matrix(svmClassifier,X_test_bow_sparse,y_test_bow,display_labels=leaveIn,cmap=plt.cm.Blues)
    print(dispBow.confusion_matrix)
    plt.show()

if True:
    bigramWordVectorizer = CountVectorizer(analyzer="word", ngram_range=(2,2))
    X_bigram_raw = bigramWordVectorizer.fit_transform(x_dataset)
    bigramWords = bigramWordVectorizer.get_feature_names()

    print(len(bigramWords))
    x_dataset_biwords = []
    for x in tqdm(x_dataset):
        word = []
        for y in bigramWords:
            word.append(x.count(y))
        x_dataset_biwords.append(word)

    print(len(x_dataset_biwords))

    x_np_words = np.array([np.array(xi) for xi in x_dataset_biwords])

    X_train_words, X_test_words, y_train_words, y_test_words = train_test_split(x_np_words, y_np, test_size=0.2, random_state=1)
    X_train_words_sparse = csr_matrix(X_train_words)
    X_test_words_sparse = csr_matrix(X_test_words)

    nbWords = MultinomialNB()
    nbWords.fit(X_train_words_sparse, y_train_words)
    y_pred_words = nbWords.predict(X_test_words_sparse)
    print("NB + 2gram Words tacnost: ",skm.accuracy_score(y_test_words,y_pred_words))
    print("SMO + 2gram Words f1 score: ",skm.f1_score(y_test_words,y_pred_words,average="micro"))
    print("SMO + 2gram Words precision: ",skm.precision_score(y_test_words,y_pred_words,average="micro"))
    print("SMO + 2gram Words recall: ",skm.recall_score(y_test_words,y_pred_words,average="micro"))

    dispWords = skm.plot_confusion_matrix(nbWords,X_test_words_sparse,y_test_words,display_labels=leaveIn,cmap=plt.cm.Blues)
    print(dispWords.confusion_matrix)
    plt.show()

if False:
    bigramCharVectorizer = CountVectorizer(analyzer="char", ngram_range=(3,3))
    X_chargram_raw = bigramCharVectorizer.fit_transform(x_dataset)
    bigramChars = bigramCharVectorizer.get_feature_names()

    print(len(bigramChars))

    x_dataset_chars = []
    for x in tqdm(x_dataset):
        char = []
        for y in bigramChars:
            char.append(x.count(y))
        x_dataset_chars.append(char)

    print(len(x_dataset_chars))

    x_np_chars = np.array([np.array(xi) for xi in x_dataset_chars])

    X_train_chars, X_test_chars, y_train_chars, y_test_chars = train_test_split(x_np_chars,y_np,test_size=0.2,random_state=1)
    X_train_chars_sparse = csr_matrix(X_train_chars)
    X_test_chars_sparse = csr_matrix(X_test_chars)

    nbChars = MultinomialNB()
    nbChars.fit(X_train_chars_sparse,y_train_chars)
    y_pred_chars = nbChars.predict(X_test_chars_sparse)
    print("NB + 3gram Chars tacnost: ",skm.accuracy_score(y_test_chars,y_pred_chars))
    print("SMO + 3gram Chars f1 score: ",skm.f1_score(y_test_chars,y_pred_chars,average="macro"))
    print("SMO + 3gram Chars precision: ",skm.precision_score(y_test_chars,y_pred_chars,average="macro"))
    print("SMO + 3gram Chars recall: ",skm.recall_score(y_test_chars,y_pred_chars,average="macro"))

    dispChars = skm.plot_confusion_matrix(nbChars,X_test_chars_sparse,y_test_chars,display_labels=leaveIn,cmap=plt.cm.Blues)
    print(dispChars.confusion_matrix)
    plt.show()
