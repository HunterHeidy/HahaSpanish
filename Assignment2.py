import codecs
import string
import re
from nltk import word_tokenize

f = codecs.open("smallcleaned.txt", encoding="UTF-8")
contents = f.read()

#the initial cleaning process seems to miss formatting between braces
#this code will remove everything between two sets of braces
contents = re.sub(r'\{\{(.*?)\}\}', "", contents)
#catch the left over links that have no closing braces
contents = re.sub(r'\{\{(.*)', "", contents)
#remove the quotes that are left over, the filter 
contents = re.sub(r'\'+', "", contents)
#remove the filenames of images but retain the title text they are called from
contents = re.sub(r'(.*)\|', "", contents)

tokens = word_tokenize(contents)

small = codecs.open('smalltokens.txt', 'w', 'utf8')
#punctuation is tokenized but they are not tokens, remove them
tokens = filter(lambda word: word not in ',-.:;()%', tokens)
for ele in tokens:
    small.write(ele+'\n')
small.close()
