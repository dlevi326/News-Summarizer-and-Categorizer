from __future__ import division, print_function  # Python 2 users only
import nltk, re
from nltk import word_tokenize
from nltk.stem import *
import os
import operator
from math import log
import pickle

class Trainer():
	def __init__(self,filelist):

		# Number of words in documents
		#self.total_docs = {"Str":0,"Pol":0,"Dis":0,"Cri":0,"Oth":0}
		self.total_words_in_doc = {}

		# Number of total words
		#self.words_per_doc = {"Str":{},"Pol":{},"Dis":{},"Cri":{},"Oth":{}}
		self.diff_words_per_doc_count = {}

		# Training file
		self.filelist = filelist


	def populate_table(self,trainfile,category):

		# Opens individual file
		f = open(trainfile,'r')

		# Tokenizes documents
		tokens = word_tokenize(f.read())

		# Stemmer
		stemmer = PorterStemmer()

		# Increases number of documents in the category

		# Adds category if not already present
		if category not in self.total_words_in_doc:
			self.total_words_in_doc[category] = 0
			self.diff_words_per_doc_count[category] = {}

		

		for token in tokens:
			#print(self.words_per_doc)

			# Stems token
			token = stemmer.stem(token)
			self.total_words_in_doc[category]+=1

			if token in self.diff_words_per_doc_count[category]:
				self.diff_words_per_doc_count[category][token]+=1
			else:
				self.diff_words_per_doc_count[category][token]=1

			
		

	def handle_files(self):

		# Opens training files
		f = open(self.filelist,"r")

		# Puts all training docs into var docs
		docs = f.readlines()

		# [filepath, category]
		os.chdir('tc_provided')
		
		for document in docs:
			self.populate_table(document.split()[0],document.split()[1])

		# Save object for later use
		#self.words_per_doc = self.diff_words_per_doc_count
		#self.total_docs = self.total_words_in_doc
		pickle.dump(self,open('./Prob.p','wb'))

