from HTMLParser import  HTMLParser
from datetime import datetime
import time
from elasticsearch import Elasticsearch
import re
from nltk.tag import pos_tag
import gensim
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import math
import nltk
from itertools import product, izip

engstop = stopwords.words('english')
set_stopword=engstop


	
			
def indexquiz(es):
	counter=0
	uniquequiz={}
	for line in open('train/bigquiz.txt'):
		line=' '.join(line.split('\t')[1:])
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		uniquequiz[line]=None
	for line in open('train/bigquiz2.txt'):
		line=' '.join(line.split('\t')[1:])
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		uniquequiz[line]=None
	for line in open('train/bigquiz3.txt'):
		line=' '.join(line.split('\t')[1:])
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		uniquequiz[line]=None
	for line in open('train/requiz.txt'):
		line=' '.join(line.split('\t')[1:])
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		uniquequiz[line]=None
	for line in open('train/requiz2.txt'):
		line=' '.join(line.split('\t')[1:])
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		uniquequiz[line]=None
	for line in open('train/requiz3.txt'):
		line=' '.join(line.split('\t')[1:])
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		uniquequiz[line]=None
	
	for line in uniquequiz:
		es.index(index="quizlets", doc_type=type, body={"text": line})
		
	
			
def indexquizlemma(es):
	counter=0
	uniquequiz={}
	for line in open('train/bigquizlemma.txt'):
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		uniquequiz[line]=None
	for line in open('train/bigquizlemma2.txt'):
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		uniquequiz[line]=None
	for line in open('train/bigquizlemma3.txt'):
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		uniquequiz[line]=None
	for line in open('train/requizlemma.txt'):
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		uniquequiz[line]=None
	for line in uniquequiz:
		es.index(index="quizlets_lemma", doc_type=type, body={"text": line})
		

		
def indexqalemma(es):
	for line in open('train/CK12lemma.txt'):
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		es.index(index="qa_lemma", doc_type=type, body={"text": line})
		
def indexqa(es):
	for line in open('train/CK12clean.txt'):
		line = line.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
		line = re.sub('[^a-zA-Z0-9,. \"\']+', " ", line)
		es.index(index="qa", doc_type=type, body={"text": line})
		

	

	
def main():
	es = Elasticsearch()  # init es
	indexquizlemma(es)
	indexquiz(es)
	indexqalemma(es)
	indexqa(es)
	
	
if __name__ == '__main__':
	main()