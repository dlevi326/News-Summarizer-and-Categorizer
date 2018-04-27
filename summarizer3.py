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

		stemmer = PorterStemmer()
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

			encounteredWords = []
			for token in tokens:
				token = stemmer.stem(token)

				if token not in self.uniqueWords:
					self.uniqueWords.append(token)

				if token not in encounteredWords:
					if token in self.docsThatContainWord:
						self.docsThatContainWord[token] += 1
					else:
						self.docsThatContainWord[token] = 1

					encounteredWords.append(token)
			trainingFile.close()

		for word in self.uniqueWords:
			self.idf[word] = log(float(numDocs)/float(self.docsThatContainWord[word]))
		f.close()

		pickle.dump(self.idf,open('./idf.p','wb'))


		



	def computeSentences(self,text):
		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		generatedSentences = sent_detector.tokenize(text)
		return generatedSentences

		

	def summarize(self, infile, outfile, numsentences):
		self.infile = infile
		self.outfile = outfile
		self.numsentences = numsentences
		self.idf = pickle.load(open('./idf.p','rb'))

		ignoreWords = stopwords.words()
		ignoreWords.append(list(string.punctuation))

		f = open(self.infile)

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

		#mostImportant = []
		#for key, value in sorted(self.tfidf.iteritems(), key=lambda (v,k): (k,v), reverse=True):
		#	mostImportant.append(key)


		# Metric 1
		'''
		outputSentences = []
		for word in mostImportant:
			if len(outputSentences)>=self.numsentences:
				break
			for sent in origSentences:
				if len(outputSentences)>=self.numsentences:
					break
				if sent not in outputSentences:
					tokens = tokenizer.tokenize(sent)
					if word in tokens:
						outputSentences.append(sent)
		'''

		
		NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
		sentRank = {}
		sentWeight = {}
		totalWeightOfDocument = 0

		for sent in origSentences:
			weight = 0
			
			tokens = tokenizer.tokenize(sent)
			POS = nltk.pos_tag(tokens)

			
			for token in POS:
				if token[1] in NOUNS:
					token = stemmer.stem(token[0])	
					#weight+=self.tfidf[token]

			sentWeight[sent] = weight
			totalWeightOfDocument+=weight

		#for sent in origSentences:
		#	sentRank[sent] = float(sentWeight[sent])/float(totalWeightOfDocument)
		
		
		
		pprint(self.uniqueWords)
		#pprint(sentRank)




	
		#pprint(mostImportant)
		#for sent in outputSentences:
		#	print sent
		#print(outputSentences)

		'''
		f = open(self.outfile,"w")
		f.write('Article Summary -- '+str(numsentences)+' sentences -- By: David Levi\n\n')
		for out in outputSentences:
			f.write(out.encode('utf-8')+'\n')
		f.close()
		'''

	def readTrainFile(self):
		idf = pickle.load(open('./idf.p','rb'))
		pprint(idf)
				

if __name__ == '__main__':
	mySummarizer = Summarizer()
	#mySummarizer.train('trainfile.txt')
	#mySummarizer.readTrainFile()
	mySummarizer.summarize('infile2.txt','outfile.txt',4)











