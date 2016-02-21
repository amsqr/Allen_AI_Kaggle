from datetime import datetime
import time
import re
from nltk.tag import pos_tag
import gensim
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import math
import nltk
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold,ShuffleSplit,StratifiedShuffleSplit,KFold
from sklearn.metrics import accuracy_score,log_loss
from nltk.stem import WordNetLemmatizer
import math
import itertools
import pickle
import string
from difflib import SequenceMatcher
from sklearn import preprocessing


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
	


categordict={'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}

def categorquestion(q,a):
	#A: Direct question: What is/Which of the following/ What has/ What is/ Which / is least/ is the least + short response
	#B: Direct question: + long response
	#C: multiphrase question + short response
	#D: multiphrase question + long response
	#E: fill the _______ / no direct q + long response
	#F: fill the _______ / no direct q+ short response

	
	
	#if len(spl)==3:
	#	return 'G'
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
	
	
def biranking():
	
	trainings = [open("data/training_set.tsv", 'r'),open("data/aristo.csv", 'r')]
	
	
	abcd = ["A", "B", "C", "D"]
	resdict={"A":0, "B":1, "C":2, "D":3}
	good=0.0
	total=0.0
	count = 0
	
	X_tuplesheur=[]
	Xheur=[]
	i=0
	idf = defaultdict(float)
	questiontypes=[]
	
	if 1==1:
		n=0
		
		for training in trainings:
			for line in training:
				count += 1
				block = line.split("\t")
				if block[0] == 'id': continue
				question=block[1]
				ans=block
				resu=ans[2]
				#print resu
				i+=1
				#print i
				localX=[]

				
				for answer_i in range(len(ans[3:7])):
					answer=ans[3:7][answer_i]

					elasticfeat=[0,0,0]
					
					if answer.lower().find("all of the above")>-1:
						elasticfeat[0]=1
					else:
						if answer.lower().find("none of the above")>-1:
							elasticfeat[1]=1
						else:
							if answer.lower().startswith('both '):
								elasticfeat[2]=1

					Xheur.append(elasticfeat)
					localX.append(elasticfeat)
						
				category=categorquestion(block[1],block[3])
				questiontypes.append(categordict[category])
				X_tuplesheur.append(localX)
			
	Xheur=np.array(Xheur)
	
	X,y,X_tuples,y_tuples=pickle.load(open('trainelasticw2v.pick','rb'))
	X2,y,X_tuples2,y_tuples=pickle.load(open('trainelasticLMJM.pick','rb'))
	#X2,y,X_tuples2,y_tuples=pickle.load(open('trainelasticBM25.pick','rb'))
	
	#X=X[0:2500*4]
	#Xheur=Xheur[0:2500*4]
	#y=y[0:2500*4]
	#X_tuples=X_tuples[0:2500]
	#y_tuples=y_tuples[0:2500]
	
	#X2=X2[0:2500*4]
	#X_tuples2=X_tuples2[0:2500]
	#X3=X3[0:2500*4]
	#X_tuples3=X_tuples3[0:2500]

	
	
	print X.shape,y.shape, Xheur.shape
	X=np.concatenate([Xheur,X,X2],axis=1)
	print X.shape,y.shape
	
	print np.array(X_tuples).shape
	
	if 1==1:
		for xxt in range(len(X_tuples)):
			#p=X_tuples[xxt]
			p2=X_tuplesheur[xxt]
			p=list(np.array(X_tuples[xxt])[:,[0,1,2,4,6,7]])
			#print p
			#p3=list(np.array(X_tuples2[xxt])[:,[0,1]])
			p3=X_tuples2[xxt]
			#p4=X_tuples3[xxt]
			
			mnp=[]
			for r in range(4):
				m=list(p[r])
				m.extend(p2[r])
				m.extend(p3[r])
				#m.extend(p4[r])
				
				mnp.append(m)
			
			X_tuples[xxt]=mnp
		
	X_tuples=np.array(X_tuples)
	y_tuples=np.array(y_tuples)
	
	X_tuples_f=[]
	y_tuples_f=[]
	
	if 1==2:
		c=-1
		for s,yi in zip(X_tuples,y_tuples):
			c+=1
			if questiontypes[c]==0:
				X_tuples_f.append(s)
				y_tuples_f.append(yi)
		
		X_tuples=np.array(X_tuples_f)
		y_tuples=np.array(y_tuples_f)
	
	print X_tuples.shape,y_tuples.shape
	skf = KFold(len(y_tuples),5)
	#skf = StratifiedKFold(y_tuples,5)
	#skf =ShuffleSplit(len(y_tuples), n_iter=10, test_size=.25)
	#skf=StratifiedShuffleSplit(y_tuples, 10, test_size=0.25, random_state=0)
	finalscores=[]

	for train, test in skf:

		newtrain2=[]
		newy2=[]
		

		
		for xtup,ytup in zip(X_tuples[train],y_tuples[train]):
			for a,b in itertools.product([0,1,2,3],[0,1,2,3]):
				if a!=b:
					
					
					nw2=xtup[a]
					nw2=np.concatenate([nw2,xtup[b]],axis=1)
					newtrain2.append(nw2)
					mycl=0
					if a==ytup:
						mycl=0
					else:
						if b==ytup:
							mycl=1
						else:
							mycl=2
					
					newy2.append(mycl)
							
		newtrain2=np.array(newtrain2)
		newy2=np.array(newy2)

		model2=LogisticRegression(C=2.0)
		model2.fit(newtrain2,newy2)
	
		
		myres=[]
		for xtup,ytup in zip(X_tuples[test],y_tuples[test]):
			ranko=[0,0,0,0]
			for a,b in itertools.product([0,1,2,3],[0,1,2,3]):
				if a!=b:
					nw2=xtup[a]
					nw2=np.concatenate([nw2,xtup[b]],axis=1)
					nw2=nw2.reshape(1, -1)
					#print nw2.shape
					res=model2.predict_proba(nw2)
					
					ranko[a]+=res[0][0]
					ranko[b]+=res[0][1]
					ranko[a]-=res[0][2]
					ranko[b]-=res[0][2]
					
			myres.append(ranko.index(max(ranko)))
		
		gold=np.array(y_tuples[test])
		res=accuracy_score(gold,myres)
		finalscores.append(res)
		print res,'3c'
	if len(finalscores)>0:
		print 'avg.',sum(finalscores)/len(finalscores)*1.0
		

	
def submitbiranking():
	
	abcd = ["A", "B", "C", "D"]
	resdict={"A":0, "B":1, "C":2, "D":3}
	good=0.0
	total=0.0
	count = 0
	
	
	X=[]
	y=[]
	
	X_tuples=[]
	y_tuples=[]
	
	i=0
	trainings = [open("data/training_set.tsv", 'r'),open("data/aristo.csv", 'r')]
	
	Xheur=[]
	X_tuplesheur=[]
	for training in trainings:
		for line in training:
			count += 1
			block = line.split("\t")
			if block[0] == 'id': continue
			question=block[1]
			ans=block
			resu=ans[2]
			#print resu
			i+=1
			#print i
			localX=[]
			#print block
			
			for answer_i in range(len(ans[3:7])):
				answer=ans[3:7][answer_i]
				
				elasticfeat=[0,0,0]
				if answer.lower().find("all of the above")>-1:
					elasticfeat[0]=1
				else:
					if answer.lower().find("none of the above")>-1:
						elasticfeat[1]=1
					else:
						if answer.lower().startswith('both '):
							elasticfeat[2]=1
				
				Xheur.append(elasticfeat)
				localX.append(elasticfeat)
					
			
			X_tuplesheur.append(localX)
			
	Xheur=np.array(Xheur)

	X,y,X_tuples,y_tuples=pickle.load(open('trainelasticw2v.pick','rb'))
	X2,y,X_tuples2,y_tuples=pickle.load(open('trainelasticLMJM.pick','rb'))
	#X2,y,X_tuples2,y_tuples=pickle.load(open('trainelasticBM25.pick','rb'))
	
	#X=X[0:2500*4]
	#y=y[0:2500*4]
	#X_tuples=X_tuples[0:2500]
	#y_tuples=y_tuples[0:2500]
	#X2=X2[0:2500*4]
	#X_tuples2=X_tuples2[0:2500]
	
	print X.shape,y.shape, Xheur.shape
	X=np.concatenate([Xheur,X,X2],axis=1)
	print X.shape,y.shape
	
	print np.array(X_tuples).shape
	
	if 1==1:
		for xxt in range(len(X_tuples)):
			#p=X_tuples[xxt]
			p2=X_tuplesheur[xxt]
			p3=X_tuples2[xxt]
			p=list(np.array(X_tuples[xxt])[:,[0,1,2,4,6,7]])
			mnp=[]
			for r in range(4):
				m=list(p[r])
				m.extend(p2[r])
				m.extend(p3[r])
				mnp.append(m)
			
			X_tuples[xxt]=mnp
		
	X_tuples=np.array(X_tuples)
	y_tuples=np.array(y_tuples)
	
	print X_tuples.shape,y_tuples.shape


	#-------------------------
	newtrain2=[]
	newy2=[]

	for xtup,ytup in zip(X_tuples,y_tuples):
		for a,b in itertools.product([0,1,2,3],[0,1,2,3]):
			if a!=b:
				
				nw2=xtup[a]
				nw2=np.concatenate([nw2,xtup[b]],axis=1)
				newtrain2.append(nw2)
				
				if a==ytup:
					newy2.append(0)
				else:
					if b==ytup:
						newy2.append(1)
					else:
						newy2.append(2)
						
	newtrain2=np.array(newtrain2)
	newy2=np.array(newy2)
	#print newtrain2.shape,newy2.shape
	model2=LogisticRegression(C=2.0)
	model2.fit(newtrain2,newy2)
	
	
	featureset=[]
	t="data/validation_set.tsv"
	o="LMJMsub.csv"
	#o="BM25sub.csv"
	output = open(o, 'w')
	count=-1
	
	X_tuplesheur=[]
	Xheur=[]
	
	input = open(t, 'r')
	for line in input:
		
		block = line.split("\t")
		if block[0] == 'id':
			continue
		question=block[1]
		ans=block
		
		localX=[]
		
		for answer_i in range(len(ans[2:6])):
			answer=ans[2:6][answer_i]
			
			elasticfeat=[0,0,0]
			
			if answer.lower().find("all of the above")>-1:
				elasticfeat[0]=1
			else:
				if answer.lower().find("none of the above")>-1:
					elasticfeat[1]=1
				else:
					if answer.lower().startswith('both '):
						elasticfeat[2]=1
					
			Xheur.append(elasticfeat)
			localX.append(elasticfeat)
		X_tuplesheur.append(localX)
			
	Xheur=np.array(Xheur)
	input = open(t, 'r')
	Xtest=np.load('subelasticw2v.npy')
	Xtest2=np.load('subelasticLMJM.npy')
	#Xtest2=np.load('subelasticBM25.npy')
	
	#print np.array(Xtest).shape,print Xheur.shape
	count=-1
	indice=-1
	for line in input:
		count+=1
		
		print("\rGenerating Submission..."+str(count))
		#print Xheur.shape,np.array(Xtest).shape
		block = line.split("\t")
		question=block[1]
		ans=block
		
		if block[0] == 'id':
			output.write(",".join(["id", "correctAnswer"]))
			output.write("\n")
			continue
		ranko=[0,0,0,0]
		xtup=[]
		for r in range(4):
			indice+=1
			#t=Xtest[indice]
			#print Xtest[indice]
			t=np.array([Xtest[indice]])[:,[0,1,2,4,6,7]][0]
			#print t
			t2=Xtest2[indice]
			t=np.concatenate([t,Xheur[indice],t2],axis=1)
			#print t.shape
			xtup.append(t)
		for a,b in itertools.product([0,1,2,3],[0,1,2,3]):
			if a!=b:
				nw2=xtup[a]
				nw2=np.concatenate([nw2,xtup[b]],axis=1)
				nw2=nw2.reshape(1, -1)
				#print nw2.shape
				res=model2.predict_proba(nw2)
				#print res
				ranko[a]+=res[0][0]
				ranko[b]+=res[0][1]
				
				ranko[a]-=res[0][2]
				ranko[b]-=res[0][2]
					
		res=ranko.index(max(ranko))
		output.write(",".join([block[0],abcd[res]]))
		output.write('\n')
	output.close()
		



#(2845L, 4L, 11L) (LB 0.60125) + BM25
#0.609841827768 3c
#0.579964850615 3c
#0.629173989455 3c
#0.581722319859 3c
#0.581722319859 3c
#avg. 0.596485061511

#(2845L, 4L, 11L) (LB 0.59375) + LMJM
#0.594024604569 3c
#0.590509666081 3c
#0.623901581722 3c
#0.608084358524 3c
#0.579964850615 3c
#avg. 0.599297012302

def main():
	start = time.time()

	biranking()
	submitbiranking()
	
	print("Time elapsed: %f" % (time.time() - start))
	
if __name__ == '__main__':
	main()