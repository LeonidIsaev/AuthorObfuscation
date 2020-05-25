import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import pickle
import textdistance
from enchant.checker import SpellChecker
import re

def subdivision(text):
    text = re.sub('(?<=[ |\(]\w)\.(?=[^\(]*\))','',text)
    text = re.sub('«','',text)
    text = re.sub('»','',text)
    s = re.sub(r'\s+', ' ', text, flags=re.M)
    
    sentences = [s for s in re.split(r'(?<=[.|!|?|…|(.*)]) ', s)]
    
    return sentences

def text_process(tex):
    lemmatiser = WordNetLemmatizer()
    russian_stopwords = stopwords.words("russian")
    # 1. Removal of Punctuation Marks 
#     nopunct=[char for char in tex if char not in string.punctuation]
#     nopunct=''.join(nopunct)
    # 2. Lemmatisation 
    a=''
    i=0
    for i in range(len(tex.split())):
        b=lemmatiser.lemmatize(tex.split()[i], pos="v")
        a=a+b+' '
    # 3. Removal of Stopwords
#     return [word for word in a.split() if word.lower() not 
#             in russian_stopwords]
    return [word for word in a.split()]    
    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
#     print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0])
                                  , range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
    
def wordcloud_(x,y):
    wordcloud1 = WordCloud().generate(x) # for EAP
    print(x)
    print(y)
    plt.imshow(wordcloud1, interpolation='bilinear')
    plt.show()

    
def model_train():
    df = pd.read_csv('D:\\AuthorObfuscation\\traindata\\authors_data_flibusta.csv', delimiter=',')
    # Defining a module for Text Processing
    y_train = df['author']
    labelencoder = LabelEncoder()
    y_train = labelencoder.fit_transform(y_train)
    
    X_train = df['text']
    
    
    
    # 80-20 splitting the dataset (80%->Training and 20%->Validation)
#     X_train, X_test, y_train, y_test = train_test_split(X, y
#                                       ,test_size=0.2, random_state=1234)
    # defining the bag-of-words transformer on the text-processed corpus # i.e., text_process() declared in II is executed...
    bow_transformer=CountVectorizer(analyzer=text_process).fit(X_train)
    # transforming into Bag-of-Words and hence textual data to numeric..
    text_bow_train=bow_transformer.transform(X_train)#ONLY TRAINING DATA
    # transforming into Bag-of-Words and hence textual data to numeric..
#     text_bow_test=bow_transformer.transform(X_test)#TEST DATA
    
    # instantiating the model with Multinomial Naive Bayes..
    model = MultinomialNB()
    # training the model...
    model = model.fit(text_bow_train, y_train)
    
    print(model.score(text_bow_train, y_train))
    
#     print(model.score(text_bow_test, y_test))
    
#     # getting the predictions of the Validation Set...
#     predictions = model.predict(text_bow_test)
#     # getting the Precision, Recall, F1-Score
#     print(classification_report(y_test,predictions))
    
#     cm = confusion_matrix(y_test,predictions)
#     plt.figure()
#     plot_confusion_matrix(cm, classes=[0,1,2,3,4,5,6,7,8,9,10], normalize=True,
#                           title='Confusion Matrix')
    
    filename = 'D:\\AuthorObfuscation\\models\\MultinomialNB\\model.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    filename = 'D:\\AuthorObfuscation\\models\\MultinomialNB\\labelencoder.sav'
    pickle.dump(labelencoder, open(filename, 'wb'))
    
    filename = 'D:\\AuthorObfuscation\\models\\MultinomialNB\\bow_transformer.sav'
    pickle.dump(bow_transformer, open(filename, 'wb'))
    

def safeness_count(x_test,y_test):
    
    filename = 'D:\\AuthorObfuscation\\models\\MultinomialNB\\labelencoder.sav'
    labelencoder = pickle.load(open(filename, 'rb'))
    
    filename = 'D:\\AuthorObfuscation\\models\\MultinomialNB\\model.sav'
    model = pickle.load(open(filename, 'rb'))
    
    filename = 'D:\\AuthorObfuscation\\models\\MultinomialNB\\bow_transformer.sav'
    bow_transformer = pickle.load(open(filename, 'rb'))
    
    y_test = labelencoder.transform(y_test)
    text_bow_test = bow_transformer.transform(x_test)
    print("Safe")
    print(model.score(text_bow_test, y_test))
    
    predictions = model.predict(text_bow_test)
    # getting the Precision, Recall, F1-Score
#     print(classification_report(y_test_obf,predictions_obf))
    authors = labelencoder.inverse_transform([1,2,3,4,5,6,7,8,9,10])
    
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, classes=authors, normalize=True,
                          title='Confusion Matrix')
    
    
def model_test(obfuscation,num = 0):
    
    df = pd.read_csv('D:\\AuthorObfuscation\\testdata\\authors_data_flibusta.csv', delimiter=',')
    x_test =  list(df['text'])      
    y_test = list(df['author'])
    
    if num != 0:
        df_obf = pd.read_csv('D:\\AuthorObfuscation\\testdata\\obfuscated_data'+num+'.csv', delimiter=',')
        x_test_obf =  list(df_obf['obfuscated_text'])      
        y_test_obf = list(df_obf['author'])
    
    
    if obfuscation:
        soundness_count(x_test, x_test_obf)
        sensibleness_count(x_test_obf) 
        safeness_count(x_test_obf,y_test_obf)
    else:
        sensibleness_count(x_test) 
        safeness_count(x_test,y_test)

        
   
    
def soundness_count(x_test, x_test_obf):
    summ_text = 0
    summ_sentence = 0
    sentences_len = 0
    for i, text in enumerate(x_test):
        sentens = subdivision(text)
        sentences_len = sentences_len +  len(sentens)
        sentens_obf = subdivision(x_test_obf[i])
        for j, sent in enumerate(sentens):
            summ_sentence = summ_sentence + textdistance.overlap(sent,sentens_obf[j])         
        summ_text = summ_text + textdistance.overlap(text,x_test_obf[i])
    print("Sound")
    print("Для текстов:")
    print(summ_text/len(x_test))
    print("Для предложений:")
    print(summ_sentence/sentences_len)
        
        

def sensibleness_count(x_test):
    summ_text = 0
    sentences_len = 0
    chkr = SpellChecker("ru_RU")
    for text in x_test:
        sentences_len = sentences_len +  len(subdivision(text))
        chkr.set_text(text)
        for err in chkr:
            summ_text = summ_text + 1
    print("Sensible")
    print("Для текстов:")
    print(summ_text/len(x_test))
    print("Для предложений:")
    print(summ_text/sentences_len)