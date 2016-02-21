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
from nltk.stem import WordNetLemmatizer
import math
import itertools
import pickle
from difflib import SequenceMatcher
import string

wordnet_lemmatizer = WordNetLemmatizer()
engstop = stopwords.words('english')
set_stopword=engstop

def lemmaorstem(w):
	return wordnet_lemmatizer.lemmatize(w)
	

def procline(q):
	q=q.replace('Which of the following ','').replace('Which of these ','').replace('Which statement BEST describes ','').replace('Which statement best describes ','')
	q=q.replace(' evidence','').replace(' description','').replace(' example','').replace(' explains','').replace(' best ','').replace(' explanation','').replace(' likely','')
	q=q.replace(' describe','').replace(' correctly','').replace(' identifies','').replace(' determines','').replace(' would','').replace(' estimate','')
	q=q.replace('Were do ','').replace('If a ','').replace('In which ','').replace('Which best ','').replace('What are ', '').replace('According ', '')
	q=q.replace('What best ','').replace('What is ','').replace('What would ', '').replace('When a ','').replace('Which statement ', '').replace(' statement', '')
	q=q.replace('Why is ','').replace('Which ','').replace('Researchers ','').replace('A scientist ','').replace('A student ','')
	q = q.replace('\t',' ').replace('\n',' ').replace('\r', ' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
	q = re.sub('[^a-zA-Z0-9\- \']+', " ", q)
	q=[lemmaorstem(m)  for m in q.lower().split(' ') if m not in set_stopword and m!='']
	return ' '.join(q)
	
	
def get_vector_from_model(model, key):
	try:
		res = model[key]
	except:
		res = np.zeros(model.syn0.shape[1])
	return res
	


def cosine_similarity(v1,v2):
	"compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
	sumxx, sumxy, sumyy = 0, 0, 0
	for i in range(len(v1)):
		x = v1[i]; y = v2[i]
		sumxx += x*x
		sumyy += y*y
		sumxy += x*y
	return sumxy/math.sqrt(sumxx*sumyy)


	
def norm_word(word):
	word = word.lower().strip('?').strip('.').strip(',').strip('!').strip(':').strip(';').strip('\"').strip('\'').strip()
	return word
	


def str2regex(input):
	reg=input.replace(' ','|')
	return reg
	
def fitness(candidate,tosearch,lth):#,kbdata):
	#print candidate
	reg=re.compile(candidate)
	lenreg=len(candidate.replace('|',''))*1.0
	totalarea=0
	totalhits=0
	bestarea=0
	bestres=''
	
	for hh in tosearch:
		#hh=kbdata[hh]
		if len(hh)<lth:
			res=reg.findall(hh)
			#print res
			if len(res)>0:
				area=sum([len(p)*1.0 for p in list(set(res))])#/lenreg
				totalarea+=area
				totalhits+=1
				if area>bestarea:
					bestarea=area
					bestres=hh
					#print bestarea,bestres
	return bestarea,bestres,totalarea,totalhits
	
def getelasticfeatures(q,a,q_lemma,a_lemma,es):
	feats=[]
	predict=-1
	score=-1
	
	question = list(set([norm_word(word) for word in q.split(' ')]))
	question = list(set(question).difference(set_stopword))
	question = ' '.join(question)
		
	for i in range(len(a)):  # get the score of each combination of the answer and question
		query2 = q_lemma + ' ' + a_lemma[i]
		query = ' '.join([question, a[i]])
		query = re.sub('[^a-zA-Z0-9,\. \']+', " ", query)
		
		query=' '.join(list(set(query.split(' '))))
		query2=' '.join(list(set(query2.split(' '))))
		sc=0
		s = es.search(index='quizlets_lemma,qa_lemma', _source=False, size=10, q=query2)
		if len(s['hits']['hits'])<10:
			sc=+sum([s['hits']['hits'][f]['_score'] for f in range(len(s['hits']['hits']))])
		else:
			sc=+sum([s['hits']['hits'][f]['_score'] for f in range(10)])
		
		s = es.search(index='quizlets', _source=False, size=10, q=query)
		sc=sc+sum([s['hits']['hits'][f]['_score'] for f in range(10)])
		feats.append(sc)
		
		s = es.search(index='quizlets_lemma,qa_lemma', _source=True, size=10, q=query2)
		tosearch=[m["_source"]["text"] for m in s['hits']['hits']]
		reg=str2regex(query2)
		bestarea,bestres,totalarea,totalhits=fitness(reg,tosearch,100000000000000)
		if totalhits==0:
			feats.append(0)
		else:
			feats.append(totalarea*1.0/totalhits*1.0*len(reg.replace('|','')))
		
	return feats
	



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
	
def getw2vfeatures(q,a,model,vocab):
	feats=[]
	question_u = q
	lst_choice = a
	
	vec_q=[]
	
	vec_q = [get_vector_from_model(model, i) for i in get_entities_from_list(vocab,question_u)]
	if hasattr(vec_q, '__iter__')==False:
		vec_q=[np.zeros(model.syn1.shape[1])]
		
	fullb=[]
	fulla=np.sum(vec_q,axis=0)
	for i_choice in range(len(a)):
		choice_u =lst_choice[i_choice]
		choice_u=[m for m in choice_u if m not in question_u]
				
		vec_c_list=[]
		ents=list(set([i for i in get_entities_from_list(vocab,choice_u)]))
		vec_c = [get_vector_from_model(model, i) for i in ents]
		if hasattr(vec_c, '__iter__')==False:
			vec_c=[np.zeros(model.syn1.shape[1])]
		fullb=np.sum(vec_c,axis=0)

	sss=0
	try:
		sss=cosine_similarity(fulla,fullb)
		if math.isnan(sss):
			sss=0
	except:
		pass
	
	feats.append(sss)
	if len(feats)==0:
		return[0]
	return [sss]

	
def fuzzymatch(w,vocab):
	if w in vocab:
		return w,1.0
	spl=w.split('_')
	candidates=[]
	for m in spl:
		if m in vocab:
			candidates.extend(vocab[m])
	#if len(candidates)==0 and w[0:3] in vocab:
	#	candidates=vocab[w[0:3]]
	maxi=''
	scor=0
	for m in candidates:
		if m[0:3]==w[0:3]:
			ratio=SequenceMatcher(None, a=m,b=w).ratio()
			if ratio>0.85:
				#print m,ratio
				if ratio>scor:
					scor=ratio
					maxi=m
	return maxi,scor
	
	
def clustervocab(w):
	vocab={}
	for line in w:
		if line not in vocab:
			vocab[line]=[line]
			if line[0:3] not in vocab:
				vocab[line[0:3]]=[]
			if line not in vocab[line[0:3]]:
				vocab[line[0:3]].append(line)
			spl=line.split('_')
			if len(spl)>1:
				for m in spl:
					if m not in vocab:
						vocab[m]=[]
					if line not in vocab[m]:
						vocab[m].append(line)
	return vocab
	
	
def categorquestion(q,a):
	#A: Direct question: What is/Which of the following/ What has/ What is/ Which / is least/ is the least + short response
	#B: Direct question: + long response
	#C: multiphrase question + short response
	#D: multiphrase question + long response
	#E: fill the _______ / no direct q + long response
	#F: fill the _______ / no direct q+ short response

	if q.find('_')>-1:
		if len(a.split(' '))<4 :
			return 'F'
		else:
			return 'E'
	if q.find('.')>-1 or q.find(',')>-1:
		if len(a.split(' '))<4:
			return 'C'
		else:
			return 'D'
	if q.find('?')>-1:
		if len(a.split(' '))<4:
			return 'A'
		else:
			return 'B'
	if len(a.split(' '))<4:
		return 'F'
	else:
		return 'E'
	
	
def genfeat_train(es):
	
	#model = gensim.models.Word2Vec.load('model/word2vec_myck12_quizlet3_stem_23gram.model')
	#vocab=clustervocab(model.vocab)
	
	
	trainings = [open("data/training_set.tsv", 'r'),open("data/aristo.csv", 'r')]
	
	abcd = ["A", "B", "C", "D"]
	resdict={"A":0, "B":1, "C":2, "D":3}
	good=0.0
	total=0.0
	count = 0
	
	X_tuples=[]
	y_tuples=[]
	
	X=[]
	y=[]
	i=0

	for training in trainings:
		for line in training:
			count += 1
			block = line.split("\t")
			if block[0] == 'id': continue
			question=block[1]
			question_lemma=procline(question)
			
			category=categorquestion(block[1],block[3])
			resdi={'A':0,'B':0,'C':0,'D':0,'E':0,'F':0}
			resdi[category]=1
			category=resdi.values()
			#category=[resdi['B'],resdi['D'],resdi['E']]
			#print category
			
			ans=block
			resu=ans[2]
			#print resu
			i+=1
			print i
			localX=[]
			#print block
			for answer_i in range(len(ans[3:7])):
				answer=ans[3:7][answer_i]
				answer_lemma=procline(answer)
				
				elasticfeat=getelasticfeatures(question,[answer],question_lemma,[answer_lemma],es)
				#w2vfeat=getw2vfeatures(question_lemma.split(' '),[answer_lemma.split(' ')],model,vocab)
				#elasticfeat.extend(w2vfeat)
				#elasticfeat.extend(category)
				
				X.append(elasticfeat)
				localX.append(elasticfeat)
				if resdict[resu]==answer_i:
					y.append(1)
				else:
					y.append(0)
					
			y_tuples.append(resdict[resu])
			X_tuples.append(localX)
		
	X=np.array(X)
	y=np.array(y)
		
	pickle.dump([X,y,X_tuples,y_tuples],open('trainelasticBM25.pick','wb+'))
	print X.shape,y.shape
	
	

def genfeat_sub(es):
	
	featureset=[]
	t="data/validation_set.tsv"
	input = open(t, 'r')
	count=-1
	for line in input:
		count+=1
		print("\rGenerating ..."+str(count))
		block = line.split("\t")
		question=block[1]
		ans=block
		question_lemma=procline(question)
		
		category=categorquestion(block[1],block[3])
		resdi={'A':0,'B':0,'C':0,'D':0,'E':0,'F':0}
		resdi[category]=1
		category=resdi.values()
				
		if block[0] == 'id':
			continue
	
		maxr=0
		maxprob=0
		for answer_i in range(len(ans[2:6])):
			answer=ans[2:6][answer_i]
			answer_lemma=procline(answer)
	
			elasticfeat=getelasticfeatures(question,[answer],question_lemma,[answer_lemma],es)
			#w2vfeat=getw2vfeatures(question_lemma.split(' '),[answer_lemma.split(' ')],model2,vocab)
			#elasticfeat.extend(w2vfeat)
			#elasticfeat.extend(category)

			featureset.append(elasticfeat)

	np.save('subelasticBM25.npy',np.array(featureset))
	

def main():
	start = time.time()
	es = Elasticsearch()  # init es
	
	genfeat_train(es)
	genfeat_sub(es)
	print("Time elapsed: %f" % (time.time() - start))

if __name__ == '__main__':
	main()