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
import pickle

root = '/home/mtonutti/Documents/latent_dirchilet_allocation/'

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()
#dictionary = open("words_alpha.txt").read() #Uncomment if using the dictionary

print("Loading text(s)...")
print()

#Read texts
documents = []
titles = []

file_root = root+'NLM_500/documents'
for filename in os.listdir(file_root):
    if filename.endswith(".txt"): 
    	os.chdir(file_root)
    	text = open(filename,errors='ignore').read()
    	documents.append(text)
    	titles.append(text.split('\n', 1)[0]) #Get title of paper
    	text.lower()


os.chdir(root)

clean_documents = []
tdm = textmining.TermDocumentMatrix()


doc_count = 0
for doc in documents:
	tokens = tokenizer.tokenize(doc) 	# Tokenize each document

	tokens = [i for i in tokens if not i in en_stop] 	# Remove useless words

	#tokens = [a for a in tokens if a in dictionary] 	# Only keep words in the english dictionary (SLOW)

	tokens = [a for a in tokens if len(a) > 3] 	#Only keep words longer than 3 letters

	#printable = set(string.printable) 	# Keep only printable words 
	#filter(lambda x: x in printable, tokens)

	tokens = [p_stemmer.stem(i) for i in tokens] 	# Stem words (remove endings to find shared meanings)
	clean_documents.append(tokens)

	tkstring = ' '.join(tokens) 	# Reformat for term-document matrix
	tdm.add_doc(tkstring)	# Add processed text to term-document matrix
	doc_count += 1
	print("Finished processing document {} out of {}".format(doc_count,len(documents)))

print()
print("Creating term-document matrix...")
print()


# Create numpy array from TDM

vocab = []
r = 0
for row in tdm.rows():
	if vocab == []:
		vocab = row
		X = np.zeros([doc_count,len(vocab)])
	else:
		X[r-1,:] = row
	r+=1

X = X.astype(int)

train_test_ratio = 0.90
ntrain = len(X*train_test_ratio)
X_train = X[:ntrain,:]
X_test = X[ntrain:,:]

# Train LDA
print("Training the LDA...")
print()
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X_train)

# Display results
topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words)))

print(model.components_)
#Save model
with open("dla_papers.sav", 'wb') as ff:
	pickle.dump(model, ff)

doc_topic = model.doc_topic_

for i in range(10):
	print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))


