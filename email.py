# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 13:17:24 2017

@author: Ashwani
"""
#%%
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import timeit
from nltk import FreqDist
from nltk.util import ngrams
from wordcloud import WordCloud
import textmining #TDM

#%%
data = pd.read_csv('Dataset\\Gmail.csv', header=None)
data.columns
data.head()

#%%
body = data[5]
print(body[3])


#%%
#Clean text

def cleantext(x):
    #To lower
    x=  x.lower()
    #Fing websites (start with www till white space)
    x = re.sub(pattern= r'www.+?\s', repl=' ', string= x)
    #Find emailid
    x = re.sub(pattern= r'[\w.-]+@[\w.-]+', repl=' ', string= x)
    #replace new lines
    x = re.sub(pattern= r'\n', repl=' ', string= x)
    #replace punctuations - non alphanumeric
    x = re.sub(pattern= r'[\W]+', repl=' ', string= x)
    #Remove FWD parts
    x = re.sub(pattern= r'FW:[\w\W]+', repl=' ', string= x)
    return x



#%%
body = body.apply(lambda x: cleantext(x))

#%%

emails = body.str.cat(sep=' ')
len(emails)

#%%
start = timeit.timeit()
body_tokens = body.apply(lambda x: word_tokenize(x))
print (timeit.timeit() - start)

#%%
start = timeit.timeit()
filtered_words = body_tokens.apply(lambda x: [w for w in x if not w in set(stopwords.words('english'))])
print (timeit.timeit() - start)

#%%
ps = PorterStemmer()
start = timeit.timeit()
stemmed_words = filtered_words.apply(lambda x: [ps.stem(w) for w in x])
print (timeit.timeit() - start)

#%%

all_words = [word for word_list in stemmed_words.values for word in word_list]

fdist = FreqDist(all_words)

#%%
fdist.plot(30,cumulative=False)

#%%

wordcloud = WordCloud(stopwords=STOPWORDS,
                  background_color=color,
                  width=2500,
                  height=2000
                 ).generate(cleaned_words_tokens)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

 
#%%

def word_grams(words, min=2, max=3):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s

"""
def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])
"""
#%%

tdm = textmining.TermDocumentMatrix()
tdm.add_doc(doc1)
tdm.add_doc(doc2)
tdm.add_doc(doc3)
tdm.write_csv('matrix.csv', cutoff=1)
for row in tdm.rows(cutoff=1):
        print row
        
#%%




#%%
import email
"""
body = email.message_from_string(body[3])
if body.is_multipart():
    for payload in body.get_payload():
        print(payload.get_payload().strip())
else:
    print(body.get_payload().strip())
"""


#%%




#%%




#%%




#%%





#%%





#%%
