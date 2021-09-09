import unicodedata
from scipy.sparse.construct import rand, random
from torch.functional import Tensor
from tqdm import tqdm
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.naive_bayes import MultinomialNB
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec, KeyedVectors
import time
import torch.optim as optim
from matplotlib import style

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

if True:
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
    print("SMO + Bag of Words f1 score: ",skm.f1_score(y_test_bow,y_pred_bow,average="macro"))
    print("SMO + Bag of Words precision: ",skm.precision_score(y_test_bow,y_pred_bow,average="macro"))
    print("SMO + Bag of Words recall: ",skm.recall_score(y_test_bow,y_pred_bow,average="macro"))
    dispBow = skm.plot_confusion_matrix(svmClassifier,X_test_bow_sparse,y_test_bow,display_labels=leaveIn,cmap=plt.cm.Blues)
    print(dispBow.confusion_matrix)
    plt.show()

if True:
    bigramWordVectorizer = CountVectorizer(analyzer="word", ngram_range=(2,2),max_features=140000)
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
    print("NB + 2gram Words f1 score: ",skm.f1_score(y_test_words,y_pred_words,average="macro"))
    print("NB + 2gram Words precision: ",skm.precision_score(y_test_words,y_pred_words,average="macro"))
    print("NB + 2gram Words recall: ",skm.recall_score(y_test_words,y_pred_words,average="macro"))

    dispWords = skm.plot_confusion_matrix(nbWords,X_test_words_sparse,y_test_words,display_labels=leaveIn,cmap=plt.cm.Blues)
    print(dispWords.confusion_matrix)
    plt.show()

if True:
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
    print("NB + 3gram Chars f1 score: ",skm.f1_score(y_test_chars,y_pred_chars,average="macro"))
    print("NB + 3gram Chars precision: ",skm.precision_score(y_test_chars,y_pred_chars,average="macro"))
    print("NB + 3gram Chars recall: ",skm.recall_score(y_test_chars,y_pred_chars,average="macro"))

    dispChars = skm.plot_confusion_matrix(nbChars,X_test_chars_sparse,y_test_chars,display_labels=leaveIn,cmap=plt.cm.Blues)
    print(dispChars.confusion_matrix)
    plt.show()

if True:
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, 3)
            self.conv2 = nn.Conv1d(32, 64, 3)
            self.conv3 = nn.Conv1d(64, 128, 3)

            x = torch.rand(300).view(-1,1,300)
            self._to_linear = None
            self.convs(x)

            self.fc1 = nn.Linear(self._to_linear,512)
            self.fc2 = nn.Linear(512,7)

        def convs(self, x):
            x = F.max_pool1d(F.relu(self.conv1(x)),(2))
            x = F.max_pool1d(F.relu(self.conv2(x)),(2))
            x = F.max_pool1d(F.relu(self.conv3(x)),(2))

            if self._to_linear is None:
                self._to_linear = x[0].shape[0]*x[0].shape[1]
            return x

        def forward(self, x):
            x = self.convs(x)
            x = x.view(-1,self._to_linear)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.softmax(x,dim = 1)

    def fwd_pass(net, x, y, optimizer, loss_function, train=False):
        if train:
            net.zero_grad()
        outputs = net(x)
        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
        acc = matches.count(True)/len(matches)
        loss = loss_function(outputs, y)

        if train:
            loss.backward()
            optimizer.step()
        return acc, loss

    def train(net, train_x, train_y, test_x, test_y, optimizer, loss_function, device, model_name):
        BATCH_SIZE = 60
        EPOCHS =10
        path = "models/"+model_name+".log"
        with open(path, "a") as f:
            for epoch in tqdm(range(EPOCHS)):
                for i in range(0, len(train_x), BATCH_SIZE):
                    batch_x = train_x[i:i+BATCH_SIZE].view(-1, 1, 300).to(device)
                    batch_y = train_y[i:i+BATCH_SIZE].to(device)

                    acc, loss = fwd_pass(net, batch_x, batch_y, optimizer, loss_function, train=True)
                    if i % 40 == 0:
                        val_acc, val_loss = test(net, test_x, test_y, optimizer, loss_function, device, size=80)
                        f.write(f"{model_name},{epoch},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")

    def test(net, test_x, test_y, optimizer, loss_function, device, size=32):
        random_start = np.random.randint(len(test_x)-size)
        x, y = test_x[random_start:random_start+size], test_y[random_start:random_start+size]
        with torch.no_grad():
            val_acc, val_loss = fwd_pass(net, x.view(-1, 1, 300).to(device), y.to(device), optimizer, loss_function)
        return val_acc, val_loss

    def create_acc_loss_graph(model_name):
        style.use("ggplot")
        path = "models/"+model_name+".log"
        contents = open(path, "r").read().split("\n")

        epochs = []
        times = []
        accuracies = []
        losses = []

        val_accs = []
        val_losses = []

        for c in contents:
            if model_name in c:
                name, epoch, timestamp, acc, loss, val_acc, val_loss = c.split(",")

                epochs.append(float(epoch))
                times.append(float(timestamp))
                accuracies.append(float(acc))
                losses.append(float(loss))
                val_accs.append(float(val_acc))
                val_losses.append(float(val_loss))

        fig = plt.figure()
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (1, 0), sharex=ax1) 

        ax1.plot(epochs, accuracies, label="acc")
        ax1.plot(epochs, val_accs, label="val_acc")
        ax1.legend(loc=2)

        ax2.plot(epochs, losses, label="loss")
        ax2.plot(epochs, val_losses, label="val_loss")
        ax2.legend(loc=2)
        ax2.set(xlabel="epoch")

        at1 = plt.subplot2grid((2, 2), (0, 1))
        at2 = plt.subplot2grid((2, 2), (1, 1), sharex=at1)

        at1.plot(times, accuracies, label="acc")
        at1.plot(times, val_accs, label="val_acc")
        at1.legend(loc=2)

        at2.plot(times, accuracies, label="acc")
        at2.plot(times, val_accs, label="val_acc")
        at2.legend(loc=2)
        at2.set(xlabel="time")

        plt.show()

    def sentance_to_wordlist(raw):
        words = raw.split()
        return words

    x_sentances = []
    for x in x_dataset:
        if len(x) > 0:
            x_sentances.append(sentance_to_wordlist(x))

    lang2Vec = Word2Vec(sg=1,seed=1,workers=4,vector_size=300,min_count=1,window=5,sample=1e-3)
    lang2Vec.build_vocab(x_sentances)


    X_tensors_list = []
    for x in x_sentances:
        sentance = 0
        #counter = 0
        for word in x:
            sentance += lang2Vec.wv.get_vector(word)
            #counter+=1
        #sentance /= counter
        sentancenp = np.array(sentance)
        X_tensors_list.append(sentancenp)

    X_tensors = torch.Tensor([i for i in X_tensors_list])
    print(X_tensors.shape)

    encoder = LabelBinarizer()
    encoder.fit_transform(leaveIn)
    y_hotencoded = encoder.transform(y_dataset)

    y_tensors = torch.Tensor([i for i in y_hotencoded])
    # print("hot encodovano",y_hotencoded[25])
    # print("dataset",y_dataset[25])
    # print("tensor",y_tensors[25])

    # val_size = int(len(X_tensors)*0.2)
    # X_train_tensors = X_tensors[:-val_size]
    # y_train_tensors = y_tensors[:-val_size]

    # X_test_tensors = X_tensors[-val_size:]
    # y_test_tensors = y_tensors[-val_size:]

    X_train_tensors, X_devtest_tensors, y_train_tensors, y_devtest_tensors = train_test_split(X_tensors, y_tensors, test_size=0.4,random_state=1)

    X_dev_tensors, X_test_tensors, y_dev_tensors, y_test_tensors = train_test_split(X_devtest_tensors,y_devtest_tensors, test_size = 0.5, random_state=1)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")


    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    MODEL_NAME ="please_work1"
    train(net,X_train_tensors,y_train_tensors,X_dev_tensors,y_dev_tensors,optimizer,loss_function,device,MODEL_NAME)

    create_acc_loss_graph("please_work1")

    y_pred = []
    y_test_real = []
    for i in range(len(X_test_tensors)):
        test = X_test_tensors[i].view(-1,1,300).to(device)
        output = net(test)
        l = torch.argmax(output).item()
        y_pred.append(l)
        res = torch.argmax(y_test_tensors[i]).item()
        y_test_real.append(res)

    labels = ["bos","eng","hrv","nno","pol","srp","swe"]
    print("Word Embeddings + CNN tacnost: ",skm.accuracy_score(y_test_real,y_pred))
    print("Word Embeddings + CNN f1 score: ",skm.f1_score(y_test_real,y_pred,average="macro"))
    print("Word Embeddings + CNN precision: ",skm.precision_score(y_test_real,y_pred,average="macro"))
    print("Word Embeddings + CNN recall: ",skm.recall_score(y_test_real,y_pred,average="macro"))
    confMatrix = skm.confusion_matrix(y_true=y_test_real,y_pred=y_pred)
    print(confMatrix)

    disp = skm.ConfusionMatrixDisplay(confusion_matrix=confMatrix,display_labels=labels)
    disp = disp.plot(cmap=plt.cm.Blues)
    plt.show()
    #bos:0
    #eng:1
    #hrv:2
    #nno:3
    #pol:4
    #srp:5
    #swe:6