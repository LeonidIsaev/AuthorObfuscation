import pandas as pd
import re
import random

def subdivision(text):
    text = re.sub('(?<=[ |\(]\w)\.(?=[^\(]*\))','',text)
    text = re.sub('«','',text)
    text = re.sub('»','',text)
    s = re.sub(r'\s+', ' ', text, flags=re.M)
    
    sentences = [s for s in re.split(r'(?<=[.|!|?|…|(.*)]) ', s)]
    return sentences

def train_test_data():
    df = pd.read_csv('D:\\AuthorObfuscation\\parsed data\\authors_data_flibusta.csv', delimiter=',')
    text = list(df['text'])
    author = list(df['author'])

    set_author = set(author)

    author_num_sent = {}
    num_text = {}
    author_texts = {}
    for auth in set_author:
        author_num_sent[auth] = 0
        author_texts[auth] = []

    for i, text in enumerate(text):
        temp = list(subdivision(text))
        author_num_sent[author[i]] = author_num_sent[author[i]] + len(temp)
        num_text[i] = len(temp)
        author_texts[author[i]].append(temp)

    min_sent = min(list(author_num_sent.values()))
    min_text = min(list(num_text.values()))

    if min_text > 40:
        min_text = 40

    min_group =  min_sent//min_text
    if min_group > 100:
        min_group = 100

    pair = []
    for author, texts in author_texts.items():    
        counter = 0
        for text in texts:
            Search = True
            last_num = 0
            num_in_text = 0
            while Search:
                now_num = last_num
                pair.append([author,' '.join(text[now_num:now_num+min_text])])
                last_num = now_num + min_text
                counter = counter + 1
                num_in_text = num_in_text + 1
                if now_num + min_text > len(text) or counter == min_group or num_in_text == 7:
                    Search = False
            if counter == min_group:
                break
                
    random.shuffle(pair)
                             
    delim = int(len(pair)*0.8)                        
                
    df_train = pd.DataFrame(pair[:delim],
                   columns=['author','text'])
                             
    df_test  = pd.DataFrame(pair[delim:],
                   columns=['author','text'])
                             
    df_train.to_csv('D:\\AuthorObfuscation\\traindata\\authors_data_flibusta.csv',index=False)
    
    df_test.to_csv('D:\\AuthorObfuscation\\testdata\\authors_data_flibusta.csv',index=False)
        
         
        
    
    
    