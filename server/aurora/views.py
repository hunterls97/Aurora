from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Process

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer() 

import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle

import json

#this shouldnt be in views but oh well
class Learner(object):
	words = []
	classes = []
	documents = []
	ignore_words = ['?']
	training = []
	output = []

	context = {}
	dynamic_context = {}

	ERROR_THRESHOLD = 0.2
	model = None

	def tokenize(self):
		with open('C:/Users/hunter/Desktop/aurora_django/aurora/resources/intents_m.json') as json_data:
			intents = json.load(json_data)

		#print(intents)
		words = self.words = []
		classes = self.classes = []
		documents = self.documents = []
		ignore_words = self.ignore_words = []

		for intent in intents['intents']:
			if 'patterns' in intent:
				for pattern in intent['patterns']:
					word = nltk.word_tokenize(pattern)
					words.extend(word)
					documents.append((word, intent['tag']))

					if intent['tag'] not in classes:
						classes.append(intent['tag'])

		for intent in intents['intents']:
			if 'context_filter' in intent:
				word = nltk.word_tokenize(intent['context_filter'])
				words.extend(word)
				documents.append((word, intent['tag']))

				if intent['tag'] not in classes:
					classes.append(intent['tag'])

		words = [stemmer.stem(word.lower()) for word in words if word not in ignore_words]
		words = sorted(list(set(words)))

		classes = sorted(list(set(classes)))

		print (len(documents), "documents")
		print (len(classes), "classes", classes)
		print (len(words), "unique stemmed words", words)

	def train(self, epochs=1000):
		words = self.words
		classes = self.classes
		documents = self.documents
		training = self.training
		output = self.output

		output_empty = [0] * len(classes)

		for doc in documents:
			bag = []
			pattern_words = doc[0]
			pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

			for w in words:
			    bag.append(1) if w in pattern_words else bag.append(0)


			output_row = list(output_empty)
			output_row[classes.index(doc[1])] = 1
			training.append([bag, output_row])

		random.shuffle(training)
		training = np.array(training)

		train_x = list(training[:,0])
		train_y = list(training[:,1])

		tf.reset_default_graph()
		net = tflearn.input_data(shape=[None, len(train_x[0])])
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
		net = tflearn.regression(net)

		self.model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
		self.model.fit(train_x, train_y, n_epoch=epochs, batch_size=8, show_metric=True)
		self.model.save('model.tflearn')
		pickle.dump({'words':words, 
					'classes':classes, 
					'train_x':train_x, 
					'train_y':train_y}, 
					open("training_data", "wb"))

	def loadNeuralInstance(self):
		data = pickle.load(open( "training_data", "rb" ))
		words = data['words']
		classes = data['classes']
		train_x = data['train_x']
		train_y = data['train_y']

		tf.reset_default_graph()
		net = tflearn.input_data(shape=[None, len(train_x[0])])
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
		net = tflearn.regression(net)
		self.model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
		self.model.load('./model.tflearn')

	def cleanSentence(self, sentence):
		sentence_words = nltk.word_tokenize(sentence)
		sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
		return sentence_words

	def bow(self, sentence, words, context, show_details=True):
		sentence_words = self.cleanSentence(sentence)

		bag = [0] * len(words)  
		for s in sentence_words:
			print(s + '\r\n')
			for i,w in enumerate(words):
				if w == s or w == context: 
					bag[i] = 1
					if show_details:
						print("found in bag: %s" % w)

		print('***bag***')
		print(bag)
		return(np.array(bag))

	def classify(self, sentence, context):
		model = self.model
		words = self.words
		classes = self.classes

		print(len(words))
		results = model.predict([self.bow(sentence, words, context)])[0]
		results = [[i,r] for i,r in enumerate(results) if r > self.ERROR_THRESHOLD]

		results.sort(key=lambda x: x[1], reverse=True)
		return_list = []
		for r in results:
			return_list.append((classes[r[0]], r[1]))

		return return_list

	def trigger(self, params):
		print(params)
		tag = params['tag']
		userID = params['userID']
		#dynamic_context = self.dynamic_context
		context = self.context
		print(context)

		#context[userID] = [] if not userID in context else context[userID]

		with open('C:/Users/hunter/Desktop/aurora_django/aurora/resources/intents_m.json') as json_data:
			intents = json.load(json_data)

		context[userID] = self.dynamic_context[tag]
		for i in intents['intents']:
			if i['tag'] == context[userID]:
				if "dynamic_context" in i:
					self.dynamic_context = i['dynamic_context']
				if "context_set" in i:
					context[userID] = i["context_set"]

				return i['responses']
		
		print(self.context[userID])
		return ['no responses found']

	def linear(self, params):
		path = params['path']

		print(path)

		with open('C:/Users/hunter/Desktop/aurora_django/aurora/resources/intents_m.json') as json_data:
			intents = json.load(json_data)

		for i in intents['intents']:
			if i['tag'] == path:
				return i['responses']


	def response(self, sentence, userID='Aurora_User_1', show_details=False):
		print('current context')
		print(self.dynamic_context)

		context = self.context
		print('context')
		print(context)

		context[userID] = [] if not userID in context else context[userID]

		with open('C:/Users/hunter/Desktop/aurora_django/aurora/resources/intents_m.json') as json_data:
			intents = json.load(json_data)

		results = self.classify(sentence, context[userID])

		print('results')
		print(results)

		if results:
			while results:
				for i in intents['intents']:
					if i['tag'] == results[0][0]:
						print(context[userID])
						if (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]) or not 'context_filter' in i:
							if 'dynamic_context' in i:
								print('setting dynamic context')
								self.dynamic_context = i['dynamic_context']
								print(self.dynamic_context)
							if 'context_set' in i:
								print('context set: ' + str(i['context_set']))
								context[userID] = i['context_set']
							else:
								context[userID] = []

							print('tag:' + str(i['tag']))

							if 'context_function' in i:
								t = i['args']
								t.update({'userID': userID})
								func = self.context_functions[i['context_function']]

								print('conteks')
								print(context)
								return random.choice(func(t))
								#return random.choice(func(self, i['tag'], userID))

							return random.choice(i['responses'])
						'''
						elif not 'context_filter' in i:
							if 'context_set' in i:
								print('context set: ' + str(i['context_set']))
								context[userID] = i['context_set']
							else:
								context[userID] = []

							if 'context_function' in i:
								func = self.context_functions[i['context_function']]
								return random.choice(self, func(i['responses']))

							return random.choice(i['responses'])'''

				results.pop(0)

		context[userID] = []
		return 'I\'m sorry, I am not yet smart enough to determine what you mean :('

	def clear(self):
		self.words = []
		self.classes = []
		self.documents = []
		self.ignore_words = ['?']
		self.training = []
		self.output = []

	def __init__(self, context = {}, dynamic_context = {}):
		self.context = context
		self.dynamic_context = dynamic_context
		self.context_functions = {
			'trigger': self.trigger,
			'linear': self.linear
		}

le = Learner()

@csrf_exempt
def index(request):
	le.tokenize()

	#le.train()
	le.loadNeuralInstance()

	jReq = json.loads(request.body)
	print(jReq)

	#print(le.bow("Can I schedule an appointment", le.words))
	res = le.response(str(jReq.get('query')))

	le.clear()
	print(res)

	return HttpResponse(res) #JsonResponse({'message': res})

def train(request):
	le = Learner()
	le.tokenize()
	le.train(epochs = 4000)

	return HttpResponse('Training complete')