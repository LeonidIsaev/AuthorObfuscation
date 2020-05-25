from lxml import html
from lxml.html import fromstring
from itertools import cycle
import traceback
import requests
import re
from selenium import webdriver
import pandas as pd
import zipfile, os
import re
import random
from random import choice
import time
from browsermobproxy import Server

def get_proxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = set()
    for i in parser.xpath('//tbody/tr')[:10]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
            proxies.add(proxy)
    return proxies

def random_headers():
    desktop_agents = ['Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                     'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
                     'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
                     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
                     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
                     'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
                     'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                     'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']
    return {'User-Agent': choice(desktop_agents),'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}

def get_data_modernLib(authors, driver):
    
    proxies = get_proxies()
    proxy_pool = cycle(proxies)
    server = Server("D:\\browsermob-proxy-2.1.4\\bin\\browsermob-proxy")
    proxy_selenium = server.create_proxy()
    
    for author in authors:
        all_links = []
        
        while True:
            try:
                page = requests.get('https://modernlib.net/books/'+authors[author]+'/')
                break
            except:
                proxy = next(proxy_pool)
                try:
                    page = requests.get('https://modernlib.net/books/'+authors[author]+'/', headers=random_headers(), proxies={"http": proxy, "https": proxy})
                    break
                except:
                    proxy = next(proxy_pool)
                
        tree = html.fromstring(page.content)
        
        book_links = tree.xpath('//a[@title="Перейти к книге"]//@href')
        for link in book_links:
            all_links.append(link)

        for link in all_links:
            print(link)
            if link != "":
                req = 'https://modernlib.net' + link
                while True:
                    try:
                        page = requests.get(req)
                        break
                    except:
                        proxy = next(proxy_pool)
                        try:
                            page = requests.get(req, headers=random_headers(), proxies={"http": proxy, "https": proxy})
                            break
                        except:
                            proxy = next(proxy_pool)
                        
                tree = html.fromstring(page.content)
                if not tree.xpath('//a[@title="Скачать книгу в формате txt"]//@href'):
                    continue
                else:
                    while True:
                        try:
                            driver.get(req)
                            break
                        except:
                            proxy_selenium = server.create_proxy()
                            chrome_options = webdriver.ChromeOptions()
                            chrome_options.add_argument("--proxy-server={0}".format(proxy_selenium.proxy))
                            chrome_options.add_argument('--ignore-certificate-errors')
                            driver.quit()
                            driver = webdriver.Chrome(chrome_options = chrome_options)

                    element = driver.find_element_by_xpath('//a[@title="Скачать книгу в формате txt"]')
                    try:
                        element.click()
                    except:
                        continue
    server.stop()
    driver.quit()   
    
def get_data_text_flibusta(authors):
    
    proxies = get_proxies()
    proxy_pool = cycle(proxies)
    
    for author in authors:
        print(author)
        all_links = []
        docnum = 0
        while True:
            try:
                page = requests.get('https://flibusta.appspot.com/a/'+authors[author]+'/')
                break
            except:
                proxy = next(proxy_pool)
                try:
                    page = requests.get('https://flibusta.appspot.com/a/'+authors[author]+'/', headers=random_headers(), proxies={"http": proxy, "https": proxy})
                    break
                except:
                    proxy = next(proxy_pool)
                
        tree = html.fromstring(page.content)
        
        book_links = tree.xpath('//a//@href')
        for link in book_links:
            if link.find('/read') != -1:
                all_links.append(link)

        for link in all_links:
            print(link)
            if link != "":
                req = 'https://flibusta.appspot.com/' + link
                while True:
                    try:
                        page = requests.get(req)
                        break
                    except:
                        proxy = next(proxy_pool)
                        try:
                            page = requests.get(req, headers=random_headers(), proxies={"http": proxy, "https": proxy})
                            break
                        except:
                            proxy = next(proxy_pool)
                        
                tree = html.fromstring(page.content)
                texts = tree.xpath('//p[@class="book"]')
                texts_data = []
                for text in texts:
                    if text.text != None:                         
                        texts_data.append(text.text)
                dir_txt = 'D:\\AuthorObfuscation\\parsed data\\Flibusta\\' + author
                if not os.path.exists(dir_txt):
                    os.makedirs(dir_txt, exist_ok=True)
                f = open(dir_txt+'\\'+str(docnum)+'.txt', 'w', encoding='utf-8')
                current_text = ' '.join(texts_data)
                current_text = (current_text).encode().decode('utf-8', 'ignore')
                f.write(current_text)
                f.close()
                docnum = docnum + 1
                
def collect_data_flibusta(author):
    data = []
    dir_txt = 'D:\\AuthorObfuscation\\parsed data\\Flibusta\\' + author
    files = os.listdir(dir_txt)
    for file in files:
        with open(dir_txt + '/' + file, encoding='utf-8') as text:
            data.append([author,clean_data(text.read())])    
    return data 

        
def get_texts_modernLib(authors):
    for author in authors:
        dir_zip = 'C:\\Users\\Леонид\\Downloads\\'
        dir_txt = 'D:\\AuthorObfuscation\\parsed data\\ModernLib\\' + author
        if not os.path.exists(dir_txt):
            os.makedirs(dir_zip, exist_ok=True) 
            os.makedirs(dir_txt, exist_ok=True)
        files = os.listdir(dir_zip)
        files = filter(lambda x: x.startswith(author), files)
        for file in files:
            fantasy_zip = zipfile.ZipFile(dir_zip + '\\' +file)
            fantasy_zip.extractall(dir_txt)
            fantasy_zip.close()
        texsts = os.listdir(dir_txt)
        urls = filter(lambda x: x.endswith('.url'), texsts)
        for url in urls:
            os.remove(dir_txt + '/' + url) 
            
def collect_data_modernLib(author):
    data = []
    dir_txt = 'D:\\AuthorObfuscation\\parsed data\\ModernLib\\' + author
    files = os.listdir(dir_txt)
    for file in files:
        with open(dir_txt + '/' + file) as text:
            mass_data = []
            for line in text.readlines():
                mass_data.append(line)
            mass_data = mass_data[7:len(mass_data)]
            mass_data = mass_data[:len(mass_data)-5]
            data.append([author,clean_data(mass_data)])     
    return data

def subdivision(text):
    text = re.sub('(?<=[ |\(]\w)\.(?=[^\(]*\))','',text)
    text = re.sub('«','',text)
    text = re.sub('»','',text)
    split_regex = re.compile('[.|!|?|…|\(*\)]')
    sentences = filter(lambda t: t, [t.strip() for t in  split_regex.split(text)])
    return sentences

def place_data_in_csv_modernLib(authors):
    for author in authors:
        df_new = pd.DataFrame(collect_data_modernLib(author),
                   columns=['author','text'])
        if os.path.exists('D:\\AuthorObfuscation\\parsed data\\authors_data_modernLib.csv'):
            df_old = pd.read_csv('D:\\AuthorObfuscation\\parsed data\\authors_data_modernLib.csv', delimiter=',')
            df_old = df_old.append(df_new)
            df_old.to_csv('D:\\AuthorObfuscation\\parsed data\\authors_data_modernLib.csv',index=False)
        else:
            df_new.to_csv('D:\\AuthorObfuscation\\parsed data\\authors_data_modernLib.csv',index=False)
    print("Writing complete")
    
def place_data_in_csv_flibusta(authors):
    for author in authors:          
        df_new = pd.DataFrame(collect_data_flibusta(author),
                   columns=['author','text'])
        if os.path.exists('D:\\AuthorObfuscation\\parsed data\\authors_data_flibusta.csv'):
            df_old = pd.read_csv('D:\\AuthorObfuscation\\parsed data\\authors_data_flibusta.csv', delimiter=',')
            df_old = df_old.append(df_new)
            df_old.to_csv('D:\\AuthorObfuscation\\parsed data\\authors_data_flibusta.csv',index=False)
        else:
            df_new.to_csv('D:\\AuthorObfuscation\\parsed data\\authors_data_flibusta.csv',index=False)
    print("Writing complete")
    
def clean_data(text):
    text = ' '.join(text.split())
    # Типовые для текстов

    text = re.sub('-{2,}', '', text)
    text = re.sub('(Р.\s?б?\s?/?No\.?\s?(\d+|\?)?)', '', text)
    # Все оставщееся
    text = re.sub('[a-zA-Z\/\\\]\|\[{}*_]', '', text)


    return text

