import os
import glob
import re
import json
import argparse
import time
import codecs
import numpy as np
import pandas as pd
import pickle
import random
from IPython.display import clear_output
from pan19_cdaa_evaluator import *
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

            
def read_files(path: str, label: str):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(path+os.sep+label+os.sep+'*.txt')
    texts=[]
    for i,v in enumerate(files):
        f=codecs.open(v,'r',encoding='utf-8')
        texts.append((f.read(),label))
        f.close()
    return texts

def regex(string: str, model: str):
    """
    Function that computes regular expressions.
    """
    string = re.sub("[0-9]", "0", string) # each digit will be represented as a 0
    string = re.sub(r'( \n| \t)+', '', string)
    #text = re.sub("[0-9]+(([.,^])[0-9]+)?", "#", text)
    string = re.sub("https:\\\+([a-zA-Z0-9.]+)?", "@", string)

    if model == 'word':
        # if model is a word n-gram model, remove all punctuation
        string = ''.join([char for char in string if char.isalnum()])

    if model == 'char-dist':
        string = re.sub("[а-яА-Я]+", "*", string)
        # string = ''.join(['*' if char.isalpha() else char for char in string])

    return string

def frequency(tokens: list):
    """
    Count tokens in text (keys are tokens, values are their corresponding frequencies).
    """
    freq = dict()
    for token in tokens:
        if token in freq:
            freq[token] += 1
        else:
            freq[token] = 1
    return freq

def represent_text(text, n: int, model: str):
    """
    Extracts all character or word 'n'-grams from a given 'text'.
    Any digit is represented through a 0.
    Each hyperlink is replaced by an @ sign.
    The latter steps are computed through regular expressions.
    """ 
    if model == 'char-std' or model == 'char-dist':

        text = regex(text, model)
        tokens = [text[i:i+n] for i in range(len(text)-n+1)] 

        if model == 'char-std' and n == 2:
            # create list of unigrams that only consists of punctuation marks
            # and extend tokens by that list
            punct_unigrams = [token for token in text if not token.isalnum()]
            tokens.extend(punct_unigrams)

    elif model == 'word':
        try:
            text = [regex(word, model) for word in text.split() if regex(word, model)]
        except:
            print(text)
        tokens = [' '.join(text[i:i+n]) for i in range(len(text)-n+1)]
    freq = frequency(tokens)

    return freq

def extract_vocabulary(texts: list, n: int, ft: int, model: str):
    """
    Extracts all character 'n'-grams occurring at least 'ft' times in a set of 'texts'.
    """
    occurrences = {}
    iter_ = 0
    for text in texts:
        text_occurrences=represent_text(text, n, model)

        for ngram in text_occurrences.keys():

            if ngram in occurrences:
                occurrences[ngram] += text_occurrences[ngram]
            else:
                occurrences[ngram] = text_occurrences[ngram]
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i] >= ft:
            vocabulary.append(i)

    return vocabulary

def extend_vocabulary(n_range: tuple, texts: list, model: str):
    n_start, n_end = n_range
    vocab = []
    for n in range(n_start, n_end + 1):
        n_vocab = extract_vocabulary(texts, n, (n_end - n) + 1, model)
        vocab.extend(n_vocab)
    return vocab

def save_model(site_name, word_range: tuple, dist_range: tuple, char_range: tuple, pt = 0.1, n_best_factor = 0.5, 
         lower = False, use_LSA = False):

    start_time = time.time()

    df_old = pd.read_csv('D:\\AuthorObfuscation\\traindata\\authors_data_'+site_name+'.csv', delimiter=',')

    train_texts =  list(df_old['text'])  
    train_labels = list(df_old['author'])
    
    

    # word n-gram vocabulary (content / semantical features)
    vocab_word = extend_vocabulary(word_range, train_texts, model = 'word')

    # character n-gram vocabulary (non-diacrictics / alphabetical symbols are distorted)
    vocab_char_dist = extend_vocabulary(dist_range, train_texts, model = 'char-dist')

    # character n-gram vocabulary (syntactical features)
    vocab_char_std = extend_vocabulary(char_range, train_texts, model = 'char-std')

    print('\t', 'language: ', 'ru')
    print('\t', len(train_texts), 'known texts')

    print('\t', 'word-based vocabulary size:', len(vocab_word))
    print('\t', 'standard character vocabulary size:', len(vocab_char_std))
    print('\t', 'non-alphabetical character vocabulary size:', len(vocab_char_dist))



    ## initialize tf-idf vectorizer for word n-gram model (captures content) ##
    vectorizer_word = TfidfVectorizer(analyzer = 'word', ngram_range = word_range, use_idf = True, 
                                      norm = 'l2', lowercase = lower, vocabulary = vocab_word, 
                                      smooth_idf = True, sublinear_tf = True)

    train_data_word = vectorizer_word.fit_transform(train_texts)
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\vectorizer_word.sav'
    pickle.dump(vectorizer_word, open(filename, 'wb'))

    train_data_word = train_data_word.toarray()

    n_best = int(len(vectorizer_word.idf_) * n_best_factor)
    idx_w = np.argsort(vectorizer_word.idf_)[:n_best]

    train_data_word = train_data_word[:, idx_w]


    ## initialize tf-idf vectorizer for char n-gram model in which non-diacritics are distorted ##

    vectorizer_char_dist = TfidfVectorizer(analyzer = 'char', ngram_range = dist_range, use_idf = True, 
                                 norm = 'l2', lowercase = lower, vocabulary = vocab_char_dist, 
                                 min_df = 0.1, max_df = 0.8, smooth_idf = True, 
                                 sublinear_tf = True)

    train_data_char_dist = vectorizer_char_dist.fit_transform(train_texts)
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\vectorizer_char_dist.sav'
    pickle.dump(vectorizer_char_dist, open(filename, 'wb'))

    train_data_char_dist = train_data_char_dist.toarray()

    n_best = int(len(vectorizer_char_dist.idf_) * n_best_factor)
    idx_c = np.argsort(vectorizer_char_dist.idf_)[:n_best]

    train_data_char_dist = train_data_char_dist[:, idx_c]


    ##  initialize tf-idf vectorizer for char n-gram model (captures syntactical features) ##
    vectorizer_char_std = TfidfVectorizer(analyzer = 'char', ngram_range = char_range, use_idf = True, 
                                 norm = 'l2', lowercase = lower, vocabulary = vocab_char_std, 
                                 min_df = 0.1, max_df = 0.8, smooth_idf = True, 
                                 sublinear_tf = True)

    train_data_char_std = vectorizer_char_std.fit_transform(train_texts)
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\vectorizer_char_std.sav'
    pickle.dump(vectorizer_char_std, open(filename, 'wb'))

    train_data_char_std = train_data_char_std.toarray()

    n_best = int(len(vectorizer_char_std.idf_) * n_best_factor)
    idx_c = np.argsort(vectorizer_char_std.idf_)[:n_best]

    train_data_char_std = train_data_char_std[:, idx_c]


    max_abs_scaler = preprocessing.MaxAbsScaler()

    ## scale text data for word n-gram model ##
    scaled_train_data_word = max_abs_scaler.fit_transform(train_data_word)
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\max_abs_scaler_word.sav'
    pickle.dump(max_abs_scaler, open(filename, 'wb'))


    ## scale text data for char dist n-gram model ##
    scaled_train_data_char_dist = max_abs_scaler.fit_transform(train_data_char_dist)
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\max_abs_scaler_char_dist.sav'
    pickle.dump(max_abs_scaler, open(filename, 'wb'))

     ## scale text data for char std n-gram model ##
    scaled_train_data_char_std = max_abs_scaler.fit_transform(train_data_char_std)
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\max_abs_scaler_char_std.sav'
    pickle.dump(max_abs_scaler, open(filename, 'wb'))

    if use_LSA:

        # initialize truncated singular value decomposition
        svd = TruncatedSVD(n_components = 63, algorithm = 'randomized', random_state = 42)    

        # Word
        scaled_train_data_word = svd.fit_transform(scaled_train_data_word)
        filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\svd_word.sav'
        pickle.dump(svd, open(filename, 'wb'))

        # Dist
        scaled_train_data_char_dist = svd.fit_transform(scaled_train_data_char_dist)
        filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\svd_char_dist.sav'
        pickle.dump(svd, open(filename, 'wb'))

        # Char
        scaled_train_data_char_std = svd.fit_transform(scaled_train_data_char_std)
        filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\svd_char_std.sav'
        pickle.dump(svd, open(filename, 'wb'))

    word = CalibratedClassifierCV(OneVsRestClassifier(SVC(C = 1, kernel = 'linear', 
                                                          gamma = 'auto')))
    word.fit(scaled_train_data_word, train_labels)
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\word.sav'
    pickle.dump(word, open(filename, 'wb'))


    char_dist = CalibratedClassifierCV(OneVsRestClassifier(SVC(C = 1, kernel = 'linear', 
                                                               gamma = 'auto')))
    char_dist.fit(scaled_train_data_char_dist, train_labels)
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\char_dist.sav'
    pickle.dump(char_dist, open(filename, 'wb'))

    char_std = CalibratedClassifierCV(OneVsRestClassifier(SVC(C = 1, kernel = 'linear', 
                                                              gamma = 'auto')))
    char_std.fit(scaled_train_data_char_std, train_labels)
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\char_std.sav'
    pickle.dump(char_std, open(filename, 'wb'))

    print('elapsed time:', time.time() - start_time)
    
    

def baseline(candidates, obfuscated, word_range: tuple, dist_range: tuple, char_range: tuple, pt = 0.1, n_best_factor = 0.5, 
         lower = False, use_LSA = False):

    start_time = time.time()
    
    if obfuscated:
        df_old = pd.read_csv('D:\\AuthorObfuscation\\testdata\\obfuscated_data.csv', delimiter=',')

        test_texts =  list(df_old['obfuscated_text'])      

        test_labels = list(df_old['author']) 
    else:
        df_old = pd.read_csv('D:\\AuthorObfuscation\\testdata\\authors_data_flibusta.csv', delimiter=',')

        test_texts =  list(df_old['text'])      

        test_labels = list(df_old['author']) 


    ## initialize tf-idf vectorizer for word n-gram model (captures content) ##
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\vectorizer_word.sav'
    vectorizer_word = pickle.load(open(filename, 'rb'))

    n_best = int(len(vectorizer_word.idf_) * n_best_factor)
    idx_w = np.argsort(vectorizer_word.idf_)[:n_best]

    test_data_word = vectorizer_word.transform(test_texts).toarray()
    test_data_word = test_data_word[:, idx_w]

    ## initialize tf-idf vectorizer for char n-gram model in which non-diacritics are distorted ##

    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\vectorizer_char_dist.sav'
    vectorizer_char_dist = pickle.load(open(filename, 'rb'))

    n_best = int(len(vectorizer_char_dist.idf_) * n_best_factor)
    idx_c = np.argsort(vectorizer_char_dist.idf_)[:n_best]

    test_data_char_dist = vectorizer_char_dist.transform(test_texts).toarray()
    test_data_char_dist = test_data_char_dist[:, idx_c]

    ##  initialize tf-idf vectorizer for char n-gram model (captures syntactical features) ##
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\vectorizer_char_std.sav'
    vectorizer_char_std = pickle.load(open(filename, 'rb'))

    n_best = int(len(vectorizer_char_std.idf_) * n_best_factor)
    idx_c = np.argsort(vectorizer_char_std.idf_)[:n_best]

    test_data_char_std = vectorizer_char_std.transform(test_texts).toarray()
    test_data_char_std = test_data_char_std[:, idx_c]

    print('\t', len(test_texts), 'unknown texts')



    ## scale text data for word n-gram model ##
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\max_abs_scaler_word.sav'
    max_abs_scaler = pickle.load(open(filename, 'rb'))

    scaled_test_data_word = max_abs_scaler.transform(test_data_word)

    ## scale text data for char dist n-gram model ##
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\max_abs_scaler_char_dist.sav'
    max_abs_scaler = pickle.load(open(filename, 'rb'))

    scaled_test_data_char_dist = max_abs_scaler.transform(test_data_char_dist)

     ## scale text data for char std n-gram model ##
    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\max_abs_scaler_char_std.sav'
    max_abs_scaler = pickle.load(open(filename, 'rb'))

    scaled_test_data_char_std = max_abs_scaler.transform(test_data_char_std)

    if use_LSA:

        # Word
        filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\svd_word.sav'
        svd = pickle.load(open(filename, 'rb'))

        scaled_test_data_word = svd.transform(scaled_test_data_word)

        # Dist
        filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\svd_char_dist.sav'
        svd = pickle.load(open(filename, 'rb'))

        scaled_test_data_char_dist = svd.transform(scaled_test_data_char_dist)

        # Char
        filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\svd_char_std.sav'
        svd = pickle.load(open(filename, 'rb'))

        scaled_test_data_char_std = svd.transform(scaled_test_data_char_std)



    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\word.sav'
    word = pickle.load(open(filename, 'rb'))

    preds_word = word.predict(scaled_test_data_word)
    probas_word = word.predict_proba(scaled_test_data_word)

    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\char_dist.sav'
    char_dist = pickle.load(open(filename, 'rb'))

    preds_dist = char_dist.predict(scaled_test_data_char_dist)
    probas_dist = char_dist.predict_proba(scaled_test_data_char_dist)

    filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\char_std.sav'
    char_std = pickle.load(open(filename, 'rb'))

    preds_char = char_std.predict(scaled_test_data_char_std)
    probas_char = char_std.predict_proba(scaled_test_data_char_std)

    # Soft Voting procedure (combines the votes of the three individual classifier)
    avg_probas = np.average([probas_word, probas_dist, probas_char], axis = 0)        
    avg_predictions = []
    for text_probs in avg_probas:
        ind_best = np.argmax(text_probs)
        avg_predictions.append(candidates[ind_best])

    # Reject option (used in open-set cases)
    count=0
    for i,p in enumerate(avg_predictions):
        sproba=sorted(avg_probas[i],reverse=True)
        if sproba[0]-sproba[1] < pt or max(sproba) < 0.25:
            avg_predictions[i]=u'<UNK>'
            count=count+1
    print('\t',count,'texts left unattributed')
    
    summ = 0
    for i in range(len(test_labels)):
        if test_labels[i] == avg_predictions[i]:
            summ = summ + 1
    
    print('accuracy: ')
 
    print(summ/len(test_labels))
            

    print('elapsed time:', time.time() - start_time)
    

# def test_model(candidates, word_range: tuple, dist_range: tuple, char_range: tuple, pt = 0.1, n_best_factor = 0.5, 
#          lower = False, use_LSA = False):
#     start_time = time.time()

#     df_old = pd.read_csv('D:\\AuthorObfuscation\\traindata\\authors_data_flibusta_to_train.csv', delimiter=',')
 
    
#     random_index = random.choices([i for i in range(len(list(df_old['text'])))],k= 48)
    
#     train_texts =  [text for i,text in enumerate(list(df_old['text'])) if i not in random_index]    
#     train_labels = [text for i,text in enumerate(list(df_old['author'])) if i not in random_index]
    
#     test_texts =  [list(df_old['text'])[i] for i in random_index]      
#     test_labels = [list(df_old['author'])[i] for i in random_index]
    
#     #word n-gram vocabulary (content / semantical features)
#     vocab_word = extend_vocabulary(word_range, train_texts, model = 'word')

#     # character n-gram vocabulary (non-diacrictics / alphabetical symbols are distorted)
#     vocab_char_dist = extend_vocabulary(dist_range, train_texts, model = 'char-dist')

#     # character n-gram vocabulary (syntactical features)
#     vocab_char_std = extend_vocabulary(char_range, train_texts, model = 'char-std')

#     print('\t', 'language: ', 'ru')
#     print('\t', len(train_texts), 'known texts')

#     print('\t', 'word-based vocabulary size:', len(vocab_word))
#     print('\t', 'standard character vocabulary size:', len(vocab_char_std))
#     print('\t', 'non-alphabetical character vocabulary size:', len(vocab_char_dist))



#     ## initialize tf-idf vectorizer for word n-gram model (captures content) ##
#     vectorizer_word = TfidfVectorizer(analyzer = 'word', ngram_range = word_range, use_idf = True, 
#                                       norm = 'l2', lowercase = lower, vocabulary = vocab_word, 
#                                       smooth_idf = True, sublinear_tf = True)

#     train_data_word = vectorizer_word.fit_transform(train_texts)
#     filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\vectorizer_word.sav'
#     pickle.dump(vectorizer_word, open(filename, 'wb'))

#     train_data_word = train_data_word.toarray()

#     n_best = int(len(vectorizer_word.idf_) * n_best_factor)
#     idx_w = np.argsort(vectorizer_word.idf_)[:n_best]

#     train_data_word = train_data_word[:, idx_w]
      
#     test_data_word = vectorizer_word.transform(test_texts).toarray()
#     test_data_word = test_data_word[:, idx_w]

#     ## initialize tf-idf vectorizer for char n-gram model in which non-diacritics are distorted ##

#     vectorizer_char_dist = TfidfVectorizer(analyzer = 'char', ngram_range = dist_range, use_idf = True, 
#                                  norm = 'l2', lowercase = lower, vocabulary = vocab_char_dist, 
#                                  min_df = 0.1, max_df = 0.8, smooth_idf = True, 
#                                  sublinear_tf = True)

#     train_data_char_dist = vectorizer_char_dist.fit_transform(train_texts)
#     filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\vectorizer_char_dist.sav'
#     pickle.dump(vectorizer_char_dist, open(filename, 'wb'))

#     train_data_char_dist = train_data_char_dist.toarray()

#     n_best = int(len(vectorizer_char_dist.idf_) * n_best_factor)
#     idx_c = np.argsort(vectorizer_char_dist.idf_)[:n_best]

#     train_data_char_dist = train_data_char_dist[:, idx_c]
  
#     test_data_char_dist = vectorizer_char_dist.transform(test_texts).toarray()
#     test_data_char_dist = test_data_char_dist[:, idx_c]

#     ##  initialize tf-idf vectorizer for char n-gram model (captures syntactical features) ##
#     vectorizer_char_std = TfidfVectorizer(analyzer = 'char', ngram_range = char_range, use_idf = True, 
#                                  norm = 'l2', lowercase = lower, vocabulary = vocab_char_std, 
#                                  min_df = 0.1, max_df = 0.8, smooth_idf = True, 
#                                  sublinear_tf = True)

#     train_data_char_std = vectorizer_char_std.fit_transform(train_texts)
#     filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\vectorizer_char_std.sav'
#     pickle.dump(vectorizer_char_std, open(filename, 'wb'))

#     train_data_char_std = train_data_char_std.toarray()

#     n_best = int(len(vectorizer_char_std.idf_) * n_best_factor)
#     idx_c = np.argsort(vectorizer_char_std.idf_)[:n_best]

#     train_data_char_std = train_data_char_std[:, idx_c]

#     test_data_char_std = vectorizer_char_std.transform(test_texts).toarray()
#     test_data_char_std = test_data_char_std[:, idx_c]
    
#     max_abs_scaler = preprocessing.MaxAbsScaler()

#     ## scale text data for word n-gram model ##
#     scaled_train_data_word = max_abs_scaler.fit_transform(train_data_word)
#     filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\max_abs_scaler_word.sav'
#     pickle.dump(max_abs_scaler, open(filename, 'wb'))
    
#     scaled_test_data_word = max_abs_scaler.transform(test_data_word)

#     ## scale text data for char dist n-gram model ##
#     scaled_train_data_char_dist = max_abs_scaler.fit_transform(train_data_char_dist)
#     filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\max_abs_scaler_char_dist.sav'
#     pickle.dump(max_abs_scaler, open(filename, 'wb'))
    
#     scaled_test_data_char_dist = max_abs_scaler.transform(test_data_char_dist)

#      ## scale text data for char std n-gram model ##
#     scaled_train_data_char_std = max_abs_scaler.fit_transform(train_data_char_std)
#     filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\max_abs_scaler_char_std.sav'
#     pickle.dump(max_abs_scaler, open(filename, 'wb'))

#     scaled_test_data_char_std = max_abs_scaler.transform(test_data_char_std)
    
#     if use_LSA:

#         # initialize truncated singular value decomposition
#         svd = TruncatedSVD(n_components = 63, algorithm = 'randomized', random_state = 42)    

#         # Word
#         scaled_train_data_word = svd.fit_transform(scaled_train_data_word)
#         filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\svd_word.sav'
#         pickle.dump(svd, open(filename, 'wb'))
        
#         scaled_test_data_word = svd.transform(scaled_test_data_word)

#         # Dist
#         scaled_train_data_char_dist = svd.fit_transform(scaled_train_data_char_dist)
#         filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\svd_char_dist.sav'
#         pickle.dump(svd, open(filename, 'wb'))
        
#         scaled_test_data_char_dist = svd.transform(scaled_test_data_char_dist)

#         # Char
#         scaled_train_data_char_std = svd.fit_transform(scaled_train_data_char_std)
#         filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\svd_char_std.sav'
#         pickle.dump(svd, open(filename, 'wb'))
        
#         scaled_test_data_char_std = svd.transform(scaled_test_data_char_std)

#     word = CalibratedClassifierCV(OneVsRestClassifier(SVC(C = 1, kernel = 'linear', 
#                                                           gamma = 'auto')))
#     word.fit(scaled_train_data_word, train_labels)
#     filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\word.sav'
#     pickle.dump(word, open(filename, 'wb'))
    
#     preds_word = word.predict(scaled_test_data_word)
#     probas_word = word.predict_proba(scaled_test_data_word)

#     char_dist = CalibratedClassifierCV(OneVsRestClassifier(SVC(C = 1, kernel = 'linear', 
#                                                                gamma = 'auto')))
#     char_dist.fit(scaled_train_data_char_dist, train_labels)
#     filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\char_dist.sav'
#     pickle.dump(char_dist, open(filename, 'wb'))
    
#     preds_dist = char_dist.predict(scaled_test_data_char_dist)
#     probas_dist = char_dist.predict_proba(scaled_test_data_char_dist)

#     char_std = CalibratedClassifierCV(OneVsRestClassifier(SVC(C = 1, kernel = 'linear', 
#                                                               gamma = 'auto')))
#     char_std.fit(scaled_train_data_char_std, train_labels)
#     filename = 'D:\\AuthorObfuscation\\models\\AuthorIdentification\\char_std.sav'
#     pickle.dump(char_std, open(filename, 'wb'))
    
#     preds_char = char_std.predict(scaled_test_data_char_std)
#     probas_char = char_std.predict_proba(scaled_test_data_char_std)

#     # Soft Voting procedure (combines the votes of the three individual classifier)
#     avg_probas = np.average([probas_word, probas_dist, probas_char], axis = 0)        
#     avg_predictions = []
#     for text_probs in avg_probas:
#         ind_best = np.argmax(text_probs)
#         avg_predictions.append(candidates[ind_best])

#     # Reject option (used in open-set cases)
#     count=0
#     for i,p in enumerate(avg_predictions):
#         sproba=sorted(avg_probas[i],reverse=True)
#         if sproba[0]-sproba[1] < pt or max(sproba) < 0.25:
#             avg_predictions[i]=u'<UNK>'
#             count=count+1
#     print('\t',count,'texts left unattributed')
    
#     summ = 0
#     for i in range(len(test_labels)):
#         if test_labels[i] == avg_predictions[i]:
#             summ = summ + 1
    
#     print('accuracy: ') 
#     print(summ/len(test_labels))
    
#     print('elapsed time:', time.time() - start_time)