# Automated News Summarizer
# By: David Levi

# Use http://newspaper-demo.herokuapp.com/ to import data

# Seki, Yohei. "Sentence Extraction by Tf Idf and Position Weighting from Newspap Er Articles." The Graduate University for Advanced Studies.

from nltk import word_tokenize
from nltk.stem import *
import nltk.data
from math import log
from nltk.corpus import stopwords
from pprint import pprint
import string
from nltk.tokenize import RegexpTokenizer
import pickle
import nltk
from Tester import Tester

class Summarizer:

	def __init__(self):
		self.infile = ''
		self.outfile = ''
		self.numsentences = 1
		self.totalWords = 0
		self.tf = {}
		self.idf = {}
		self.tfidf = {}
		self.docsThatContainWord = {}
		self.uniqueWords = []

	def train(self,trainfile):
		# Train idf values based on training corpus

		stemmer = PorterStemmer()

		# Regexp tokenizer gave better results
		tokenizer = RegexpTokenizer('\w+')

		f = open(trainfile)

		docs = f.readlines()
		docsList = []
		for document in docs:
			docsList.append(document.split()[0])
			
		numDocs = len(docsList)
		for doc in docsList:
			trainingFile = open(doc)
			text = trainingFile.read()

			tokens = tokenizer.tokenize(text)

			# Keep track of already encountered words in document
			encounteredWords = []
			for token in tokens:
				token = stemmer.stem(token)

				if token not in self.uniqueWords:
					self.uniqueWords.append(token)

				if token not in encounteredWords:
					if token in self.docsThatContainWord:

						# Add one to the # of documents that contain this word
						self.docsThatContainWord[token] += 1
					else:
						self.docsThatContainWord[token] = 1

					encounteredWords.append(token)
			trainingFile.close()

		for word in self.uniqueWords:
			self.idf[word] = log(float(numDocs)/float(self.docsThatContainWord[word]))
		f.close()

		# Dump into pickle file
		pickle.dump(self.idf,open('./idf.p','wb'))


		



	def computeSentences(self,text): 

		# Splits up sentences using nltk's built in handler
		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		generatedSentences = sent_detector.tokenize(text)
		return generatedSentences

		

	def summarize(self, infile, outfile, numsentences,headline = None,category = None):
		self.infile = infile
		self.outfile = outfile
		self.numsentences = numsentences

		# Open pickle file
		self.idf = pickle.load(open('./idf.p','rb'))

		ignoreWords = []

		f = open(self.infile)

		# Used to take care of any unknown chars in the new article (This was needed a lot)
		text = f.read().decode('utf-8')

		stemmer = PorterStemmer()

		sentences = self.computeSentences(text)
		origSentences = self.computeSentences(text)

		tokenizer = RegexpTokenizer('\w+')

		# Compute TF
		for sent in sentences:
			tokens = tokenizer.tokenize(sent)
			for token in tokens:
				token = stemmer.stem(token)

				if token not in ignoreWords:
					if token not in self.uniqueWords:
						self.uniqueWords.append(token)
					
					if token in self.tf:
						# Have seen this token already
						self.tf[token]+=1
						self.totalWords+=1
					else:
						# Havent seen this token yet
						self.totalWords+=1
						self.tf[token] = 1

		for token in self.uniqueWords:
			if token not in self.tf:
				self.tf[token] = 0
			if token not in self.idf:
				self.idf[token] = 10
			self.tfidf[token] = float(self.tf[token])*float(self.idf[token])

		# List of NLTK noun tags
		NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
		sentRank = {}
		sentWeight = {}
		totalWeightOfDocument = 0

		# Computes headline phrases
		headPhrases = 0
		headPhraseWords = []
		if(headline):
			tokens = tokenizer.tokenize(headline)
			headPOS = nltk.pos_tag(tokens)

			for token in headPOS:
				if token[1] in NOUNS:
					token = stemmer.stem(token[0])
					headPhrases+=1
					headPhraseWords.append(token)
		headWeights = {}
		for word in headPhraseWords:
			headWeights[word] = float(1)/float(headPhrases)


		# Calculating weights in sentences
		for sent in origSentences:
			weight = 0
			
			tokens = tokenizer.tokenize(sent)
			POS = nltk.pos_tag(tokens)

			flag = False
			for token in POS:
				if token[1] in NOUNS:
					token = stemmer.stem(token[0])	
					weight+=self.tfidf[token]

					if token in headPhraseWords:
						flag = True

			if flag:
				weight += (float(1)/float(headPhrases))*.01


			sentWeight[sent] = weight
			totalWeightOfDocument+=weight

		# Actually compute the ranking
		for sent in origSentences:
			sentRank[sent] = float(sentWeight[sent])/float(totalWeightOfDocument)

		self.mostImportant = []
		self.rankings = {}

		# Sorts the most important values
		for key, value in sorted(sentRank.iteritems(), key=lambda (v,k): (k,v), reverse=True):
			self.mostImportant.append(key)
			self.rankings[key] = value
		

		mostImportantModified = self.mostImportant[:self.numsentences]
		outputSentences = []

		# Outputs the most important sentences
		for sent in origSentences:
			if sent in mostImportantModified:
				outputSentences.append(sent)

		
		f = open(self.outfile,"w")
		f.write('Article Summary -- '+str(numsentences)+' sentences -- By: David Levi\n\n')
		if(category):
			if(category=='Str'):
				cat = 'Struggle'
			elif(category=='Pol'):
				cat = 'Politics'
			elif(category=='Dis'):
				cat = 'Disaster'
			elif(category=='Cri'):
				cat = 'Crime'
			else:
				cat = 'Other'
			f.write('Category: '+str(cat)+'\n\n')
		if(headline):
			f.write(str(headline)+'\n\n')
		for out in outputSentences:
			f.write(out.encode('utf-8')+'\n')
		f.close()
		

	def readTrainFile(self):
		idf = pickle.load(open('./idf.p','rb'))
		pprint(idf)

	def printMostImportant(self):
		for sent in self.mostImportant:
			print('-'*60)
			print(sent+'-->'+str(self.rankings[sent]))
			#print(sent)
			print('-'*60)
				

if __name__ == '__main__':
	mySummarizer = Summarizer()
	#mySummarizer.train('trainfile.txt')
	#mySummarizer.readTrainFile()

	headline = ''
	infile = raw_input('Please enter document you would like summarized: ')
	outfile = raw_input('Please enter document you would like the summary to be written to: ')
	sentenceNumber = raw_input('Please enter the number of sentences you would like the summary to be: ')
	category = Tester(infile)
	result = category.execute_tests()
	mySummarizer.summarize(infile,outfile,int(sentenceNumber),headline,result)
	ans = raw_input('Would you like to see the sentence rank [y/n]? ')
	if(ans=='y'):
		mySummarizer.printMostImportant()
	else:
		print('Goodbye!')











