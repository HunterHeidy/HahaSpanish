import pandas as pd
import re
import nltk
import sys
# nltk.download('stopwords')
# import nltk
# nltk.download('punkt')

#1. Load the dataset into dataframe df
df = pd.read_csv('data/haha_2019_test.csv')

import csv
import string
import wordninja
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('spanish'))
exclude = set(string.punctuation) 

#clean emojis
def clean_emoji(sen): 
    sen = ''.join(c for c in sen if c <= '\uFFFF')
    return sen.replace("  ", " ")

#further cleaning
def clean(sen,remove_stopwords = True, contraction = True, pun= True,lemma_= False):
#     re.sub(pattern, repl, string, count=0, flags=0)

# pattern：表示正则表达式中的模式字符串；

# repl：被替换的字符串（既可以是字符串，也可以是函数）；

# string：要被处理的，要被替换的字符串；

# count：匹配的次数, 默认是全部替换

# flags：具体用处不详


    sen = re.sub(r'\{\{(.*?)\}\}', "", sen)
    #catch the left over links that have no closing braces
    sen = re.sub(r'\{\{(.*)', "", sen)
    #remove the quotes that are left over, the filter 
    sen = re.sub(r'\'+', "", sen)
    #remove the filenames of images but retain the title text they are called from
    sen = re.sub(r'(.*)\|', "",sen)
    sen = sen.strip(""" '!:?-_().,'"[]{};*""")



    sen = ' '.join([w.strip(""" '!:?-_().,'"[]{};*""") for w in re.split(' ', sen)])

    sen = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", " NUMBER ", sen)
   
    # spliting words
    string = []
    for x in sen.split():
        if len(x)>6:
            for i in wordninja.split(x):#分词
                if len(i)>2:
                    string.append(i)
        else:
            string.append(x)
    sen = " ".join(string)
    
    contraction  
    new_text = []
    for word in sen.split():#切片 默认空格

        # if word in contractions:
        #     new_text.append(contractions[word])
        # else:

        new_text.append(word)
    sen = " ".join(new_text)

    sen = re.sub(r"[^A-Za-z0-9:(),\'\`]", " ", sen)
    sen = re.sub(r"\b\d+\b", "", sen)  #remove numbers 
    sen = re.sub('\s+',  ' ', sen) #matches any whitespace characte
    sen = re.sub(r'(?:^| )\w(?:$| )', ' ', sen).strip() #removing single character
   
     # Optionally, remove stop words
    if remove_stopwords:
        sen = " ".join([i for i in sen.split() if i not in stop])
       
    # Optionally emove puncuations 
    if pun:
        sen = ''.join(ch for ch in sen if ch not in exclude)
    
    # Optionally lemmatiztion  
    if lemma_:
        normalized = " ".join(WordNetLemmatizer().lemmatize(word) for word in sen.split())        
        
    return sen.strip().lower()#转成小写

# Cleaning the dataset 
clean_data = []
for index, row in df['text'].iteritems():#（）返回迭代器（index:row）
    row = clean_emoji(str(row))
    row = clean(row, remove_stopwords=False)
    #print(row)
    clean_data.append(row)

# Inspect the cleaned summaries and texts to ensure they have been cleaned well
for i in range(len(df['text'])):
    print("Clean Review #",i+1)
    print(clean_data[i])
    print()
df['text']=clean_data
df.to_csv("data/cleaned_data_test_haha.csv", index=False , encoding='utf-8')