# encoding=utf8  

import numpy as np
import lda
import lda.datasets
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import textmining
import string
import os
import sys  

root = '/home/mtonutti/Documents/latent_dirchilet_allocation/'

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

print("Loading text(s)...")
print()

documents = []

file_root = root+'NLM_500/documents'
for filename in os.listdir(file_root):
    if filename.endswith(".txt"): 
    	os.chdir(file_root)
    	text = open(filename,errors='ignore').read().lower()
    	documents.append(text)


clean_documents = []
tdm = textmining.TermDocumentMatrix()

os.chdir(root)
dictionary = open("words_alpha.txt").read()

doc_count = 0
for doc in documents:
	#Tokenize each document
	tokens = tokenizer.tokenize(doc)
	
	#Remove useless words
	tokens = [i for i in tokens if not i in en_stop]

	#Only keep words in english dictionary (SLOW)
	#tokens = [a for a in tokens if a in dictionary]

	#Only keep words longer than 3 letters
	tokens = [a for a in tokens if len(a) > 3]

	#Only printable words 
	#printable = set(string.printable)
	#filter(lambda x: x in printable, tokens)

	#Stem words (remove endings to find shared meanings)
	tokens = [p_stemmer.stem(i) for i in tokens]
	clean_documents.append(tokens)

	tkstring = ' '.join(tokens)
	tdm.add_doc(tkstring)
	doc_count += 1
	print("Finished processing document {} out of {}".format(doc_count,len(documents)))

print()
print("Creating term-document matrix...")
print()


X = np.zeros([doc_count,len(vocab)])

vocab = []
for row in tdm.rows():
	if vocab == []:
		vocab = row
	else:
		X[r-1,:] = row

X = X.astype(int)

print("Fitting the LDA...")
print()
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X)

topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {}: {}'.format(i+1, ' '.join(topic_words)))