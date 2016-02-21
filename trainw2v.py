import re
import gensim
import os
import itertools
import numpy as np
import argparse
import string
from nltk.corpus import stopwords
import glob
from bs4 import BeautifulSoup
from mycounter import Counter
import math
import pickle
from sklearn.metrics import pairwise
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher
import random


wordnet_lemmatizer = WordNetLemmatizer()
engstop = stopwords.words('english')
set_stopword=engstop
parser = argparse.ArgumentParser("Word2Vec")


def lemmaorstem(w):
	return wordnet_lemmatizer.lemmatize(w)
	
def proclineb(q):
	q=q.replace('Which of the following ','').replace('Which of these ','').replace('Which statement BEST describes ','').replace('Which statement best describes ','')
	q=q.replace(' evidence','').replace(' description','').replace(' example','').replace(' explains','').replace(' best ','').replace(' explanation','').replace(' likely','')
	q=q.replace(' describe','').replace(' correctly','').replace(' identifies','').replace(' determines','').replace(' would','').replace(' estimate','')
	q=q.replace('Were do ','').replace('If a ','').replace('In which ','').replace('Which best ','').replace('What are ', '').replace('According ', '')
	q=q.replace('What best ','').replace('What is ','').replace('What would ', '').replace('When a ','').replace('Which statement ', '').replace(' statement', '')
	q=q.replace('Why is ','').replace('Which ','').replace('Researchers ','').replace('A scientist ','').replace('A student ','')
	q = q.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
	q = re.sub('[^a-zA-Z0-9\- \']+', " ", q)
	q=[lemmaorstem(m)  for m in q.lower().split(' ') if m not in set_stopword and m!='']
	return q#' '.join(q)
	
	
	
def procline(q,stem=True):
	q=q.replace('Which of the following ','').replace('Which of these ','').replace('Which statement best describes ','')
	q=q.replace('Were do ','').replace('If a ','').replace('In which ','').replace('Which best ','').replace('What are ', '')
	q=q.replace('What best ','').replace('What is ','').replace('What would ', '').replace('When a ','').replace('Which statement ', '')
	q=q.replace('Why is ','').replace('Which ','')
	q = q.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
	q = re.sub('[^a-zA-Z0-9\- \']+', " ", q)
	if stem==False:
		q=[m  for m in q.lower().split(' ') if m not in set_stopword and m!='' and m!=' ']
	else:
		q=[lemmaorstem(m)  for m in q.lower().split(' ') if m not in set_stopword and m!='' and m!=' ']
	return q
	

		
	
def get_vector_from_model(model, key):
	if key in model:
		return model[key]
	else:
		#return np.random.uniform(-0.25,0.25,model.syn0.shape[1]) 
		return  np.zeros(model.syn0.shape[1])
	
			
def get_entities_from_list(vocab,list):
	lista=[]
	for i in range(len(list)-1):
		if i<len(list):
			#print i,list,len(list)
			word1 = list[i]+'_'+list[i+1]
			#print word1
			if word1 in vocab:
				lista.append(word1)
			word2 = list[i+1]+'_'+list[i]
			#print word2
			if word2 in vocab:
				lista.append(word2)
			else:
				
				w,scor=fuzzymatch(word1,vocab)
				#print w,word1,scor
				if w!='' and w.find('_')!=-1:
					lista.append(w)
				else:
					w,scor=fuzzymatch(word2,vocab)
					#print w,word2,scor
					if w!='' and w.find('_')!=-1:
						lista.append(w)
	#print lista
	final=[]
	for m in range(len(list)):
		if list[m] in vocab:
			lista.append(list[m])
		else:
			w,scor=fuzzymatch(list[m],vocab)
			if w!='':
				lista.append(w)
	#print lista
	return lista
	


	
def trainw2v2():
	sents=[]
	if 1==2:
		sents2=[]
		for line in open('train/CK12clean.txt'):
			#text = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
			#text = re.sub('[^a-zA-Z0-9 \']+', " ", text.lower()).replace('  ',' ')
			#sent= [m for m in text.split(' ') if m not in engstop and m!='' and m!=' ']
			sent=procline(line)
			sents.append(sent)
			sents2.append(' '.join(sent))
			
		f=open('CK12lemma.txt','wb+')
		f.write('\n'.join(sents2))
		f.close()
		
		lines={}
		lines2=[]
		for line in open('train/bigquiz.txt'):
			#text = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
			#text = re.sub('[^a-zA-Z0-9 \']+', " ", text.lower()).replace('  ',' ')
			#sent= [m for m in text.split(' ') if m not in engstop and m!='' and m!=' ']
			line=line.split('\t')
			line=' '.join(line[1:])
			lines[line]=None
			
		for line in lines:
			sent=procline(line)
			lines2.append(' '.join(sent))
			sents.append(sent)
	if 1==2:
		lines={}
		lines2=[]
		for line in open('train/bigquiz2.txt'):
			#text = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
			#text = re.sub('[^a-zA-Z0-9 \']+', " ", text.lower()).replace('  ',' ')
			#sent= [m for m in text.split(' ') if m not in engstop and m!='' and m!=' ']
			line=line.split('\t')
			line=' '.join(line[1:])
			lines[line]=None
			
		for line in lines:
			sent=procline(line)
			lines2.append(' '.join(sent))
			sents.append(sent)
		f=open('bigquizlemma2.txt','wb+')
		f.write('\n'.join(lines2))
		f.close()
		
		lines={}
		lines2=[]
		for line in open('train/bigquiz3.txt'):
			#text = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
			#text = re.sub('[^a-zA-Z0-9 \']+', " ", text.lower()).replace('  ',' ')
			#sent= [m for m in text.split(' ') if m not in engstop and m!='' and m!=' ']
			line=line.split('\t')
			line=' '.join(line[1:])
			lines[line]=None
			
		for line in lines:
			sent=procline(line)
			lines2.append(' '.join(sent))
			sents.append(sent)
		f=open('bigquizlemma3.txt','wb+')
		f.write('\n'.join(lines2))
		f.close()
		
	if 1==1:
		bidict,tridict = pickle.load(open('quizletvocab3.pick','rb'))
		lines={}
		for line in open('train/bigquizlemma.txt'):
			lines[line.rstrip()]=None
		for line in open('train/bigquizlemma2.txt'):
			lines[line.rstrip()]=None
		for line in open('train/bigquizlemma3.txt'):
			lines[line.rstrip()]=None
		for line in open('train/CK12lemma.txt'):
			lines[line.rstrip()]=None
		for line in lines:
			line=line.split(' ')
			line.extend(get_entities_from_list(bidict,line))
			line.extend(get_entities_from_list(tridict,line))
			sents.append(line)
			#print line
		lines=None
	path_model = 'model/word2vec_myck12_quizlet3_stem_23gram.model'
	model = gensim.models.Word2Vec(sents, min_count = 5, workers = 4, size = 300, window = 9, iter = 10)
	##model = gensim.models.Word2Vec(sents, sg=0,min_count = 5, workers = 4, size = 300, window = 5, iter = 10)
	model.save(path_model)
	
	
	#f=open('bigquizlemma.txt','wb+')
	#f.write('\n'.join(lines2))
	#f.close()


def fuzzymatch(w,vocab):
	if w in vocab:
		return w,1.0
	spl=w.split('_')
	candidates=[]
	for m in spl:
		if m in vocab:
			candidates.extend(vocab[m])
	if len(candidates)==0 and w[0:3] in vocab:
		candidates=vocab[w[0:3]]
	maxi=''
	scor=0
	if isinstance(candidates, list)==False:
		return maxi,scor
	for m in candidates:
		if m[0:3]==w[0:3]:
			ratio=SequenceMatcher(None, a=m,b=w).ratio()
			if ratio>0.85:
				#print w,m,ratio
				if ratio>scor:
					scor=ratio
					maxi=m
	return maxi,scor
	
trainw2v2()