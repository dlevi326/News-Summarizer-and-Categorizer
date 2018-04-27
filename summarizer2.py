from nltk import word_tokenize
from nltk.stem import *
import nltk.data
from math import log
from nltk.corpus import stopwords
from pprint import pprint
import string
from nltk.tokenize import RegexpTokenizer

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

	def computeSentences(self,text):
		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		generatedSentences = sent_detector.tokenize(text)
		return generatedSentences

		

	def summarize(self, infile, outfile, numsentences):
		self.infile = infile
		self.outfile = outfile
		self.numsentences = numsentences

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

		# Compute IDF
		numTotalSentences = len(sentences)
		for sent in sentences:
			wordsSeen=[]
			tokens = tokenizer.tokenize(sent)
			for token in tokens:
				token = stemmer.stem(token)

				if token not in ignoreWords:
					if token not in wordsSeen:
						if token in self.docsThatContainWord:
							self.docsThatContainWord[token]+=1
						else:
							self.docsThatContainWord[token]=1
					wordsSeen.append(token)


		for token in self.uniqueWords:
			self.tfidf[token] = float(self.tf[token])*(float(log(float(numTotalSentences)/float(self.docsThatContainWord[token]))))

		mostImportant = []
		for key, value in sorted(self.tfidf.iteritems(), key=lambda (v,k): (k,v), reverse=True):
			mostImportant.append(key)

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

		#pprint(mostImportant)
		#for sent in outputSentences:
		#	print sent
		#print(outputSentences)

		f = open(self.outfile,"w")
		f.write('Article Summary -- '+str(numsentences)+' sentences -- By: David Levi\n\n')
		for out in outputSentences:
			f.write(out.encode('utf-8')+'\n')
		f.close()

				

if __name__ == '__main__':
	mySummarizer = Summarizer()
	mySummarizer.summarize('infile.txt','outfile.txt',4)











