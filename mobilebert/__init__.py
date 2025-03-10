import numpy as np
import tensorflow as tf
import bert.tokenization as bt
import math
import time
from time import perf_counter

import torch


class MobileBERT:
	def __init__(self, filename):
		self.max_length = 384
		self.tokenizer = bt.FullTokenizer(__file__.replace("__init__.py","vocab.txt"), True)
		with tf.device('/GPU:0'):
			self.interpreter = tf.lite.Interpreter(model_path=__file__.replace("__init__.py", filename))
			self.interpreter.allocate_tensors()
			self.input_details = self.interpreter.get_input_details()
			self.output_details = self.interpreter.get_output_details()

	def get_summary(self):
		print("Inputs:",self.input_details,"\nOutputs:",self.output_details)

	def get_masks(self,tokens):
		if len(tokens)>self.max_length:
			raise IndexError("Token length more than max seq length!", tokens)
		return np.asarray([1]*len(tokens) + [0] * (self.max_length - len(tokens)))

	def get_segments(self,tokens):
		if len(tokens)>self.max_length:
			raise IndexError("Token length more than max seq length!")
		segments = []
		current_segment_id = 0
		for token in tokens:
			segments.append(current_segment_id)
			if token == "[SEP]":
				current_segment_id = 1
		return np.asarray(segments + [0] * (self.max_length - len(tokens)))

	def get_ids(self,tokens):
		token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		input_ids = token_ids + [0] * (self.max_length-len(token_ids))
		return np.asarray(input_ids)

	def compile_text(self,text):
		text = text.lower().replace("-"," ")
		return ["[CLS]"] + self.tokenizer.tokenize(text) + ["[SEP]"]

	def getEmbedding(self,string):
		with tf.device('/GPU:0'):
			stokens =  self.compile_text(string)
			if len(stokens)>self.max_length:
				raise IndexError("Token length more than max seq length!")
				print("Max exceeded")
			input_ids = tf.dtypes.cast(self.get_ids(stokens),tf.int32)
			input_masks = tf.dtypes.cast(self.get_masks(stokens),tf.int32)
			input_segments = tf.dtypes.cast(self.get_segments(stokens),tf.int32)

			self.interpreter.set_tensor(self.input_details[0]['index'], [input_ids])
			self.interpreter.set_tensor(self.input_details[1]['index'], [input_masks])
			self.interpreter.set_tensor(self.input_details[2]['index'], [input_segments])
			self.interpreter.invoke()
			end_logits = self.interpreter.tensor(self.output_details[0]['index'])()
			start_logits = self.interpreter.tensor(self.output_details[1]['index'])()
		return {"start":start_logits,"end":end_logits}

	def run(self,query,context):
		with tf.device('/GPU:0'):
			stokens =  self.compile_text(query) + self.compile_text(context)
			if len(stokens)>self.max_length:
				raise IndexError("Token length more than max seq length!")
				print("Max exceeded")

			input_ids = tf.dtypes.cast(self.get_ids(stokens),tf.int32)
			input_masks = tf.dtypes.cast(self.get_masks(stokens),tf.int32)
			input_segments = tf.dtypes.cast(self.get_segments(stokens),tf.int32)

			self.interpreter.set_tensor(self.input_details[0]['index'], [input_ids])
			self.interpreter.set_tensor(self.input_details[1]['index'], [input_masks])
			self.interpreter.set_tensor(self.input_details[2]['index'], [input_segments])
			self.interpreter.invoke()
			end_logits = self.interpreter.tensor(self.output_details[0]['index'])()
			start_logits = self.interpreter.tensor(self.output_details[1]['index'])()
			end = tf.math.argmax(end_logits,output_type=tf.dtypes.int32,axis=1).numpy()[0]
			start = tf.math.argmax(start_logits,output_type=tf.dtypes.int32,axis=1).numpy()[0]
		answers = " ".join(stokens[start:end+1]).replace("[CLS]","").replace("[SEP]","").replace(" ##","")
		return answers

	def run4example(self,example,m2=None):
		device="cpu"
		desired_string = "CPU"
		if m2!= None:
			device = str(m2.device)
			desired_string = ""
			if device == "cpu":
				desired_string = "CPU"
			else:
				device = "cuda"
				desired_string = "GPU"
		with tf.device('/'+desired_string+':0'):
			# print('****************************************************')

			ent = example.features.feature['tokens']
			stokens = []
			for val in ent.bytes_list.value:
				stokens.append(val.decode(encoding='utf-8'))
			
			# print(stokens)
			# print('****************************************************')
			if len(stokens)>self.max_length:
				raise IndexError("Token length more than max seq length!")
				print("Max exceeded")

			input_ids = tf.dtypes.cast(self.get_ids(stokens),tf.int32)
			input_masks = tf.dtypes.cast(self.get_masks(stokens),tf.int32)
			input_segments = tf.dtypes.cast(self.get_segments(stokens),tf.int32)

			self.interpreter.set_tensor(self.input_details[0]['index'], [input_ids])
			self.interpreter.set_tensor(self.input_details[1]['index'], [input_masks])
			self.interpreter.set_tensor(self.input_details[2]['index'], [input_segments])

			if m2==None:
				st = perf_counter()
				self.interpreter.invoke()
				tm = perf_counter() - st
				end_logits = self.interpreter.tensor(self.output_details[0]['index'])()
				start_logits = self.interpreter.tensor(self.output_details[1]['index'])()
				end = tf.math.argmax(end_logits, output_type=tf.dtypes.int32, axis=1).numpy()[0]
				start = tf.math.argmax(start_logits, output_type=tf.dtypes.int32, axis=1).numpy()[0]
			else:
				torch_ids = self.get_ids(stokens)
				torch_ids = np.expand_dims(torch_ids,axis=0)
				torch_masks = self.get_masks(stokens)
				torch_masks = np.expand_dims(torch_masks, axis=0)
				torch_segments = self.get_segments(stokens)
				torch_segments = np.expand_dims(torch_segments, axis=0)

				torch_ids = torch.IntTensor(torch_ids)
				torch_ids = eval("torch_ids." + device + "()")
				torch_masks = torch.IntTensor(torch_masks)
				torch_masks = eval("torch_masks." + device + "()")
				torch_segments = torch.IntTensor(torch_segments)
				torch_segments = eval("torch_segments." + device + "()")

				st = perf_counter()
				res = m2(torch_ids,torch_masks,torch_segments)
				tm = perf_counter() - st
				start_logits = np.asarray(res['start_logits'].cpu().detach())
				end_logits = np.asarray(res['end_logits'].cpu().detach())
				end = tf.math.argmax(end_logits,output_type=tf.dtypes.int32,axis=1).numpy()[0]
				start = tf.math.argmax(start_logits,output_type=tf.dtypes.int32,axis=1).numpy()[0]
		tokens = " ".join(stokens).replace("[CLS]", "").replace("[SEP]", "").replace(" ##", "")
		answers = " ".join(stokens[start:end + 1]).replace("[CLS]", "").replace("[SEP]", "").replace(" ##", "")
		return tokens, answers, tm,

	def square_rooted(self,x):
		return math.sqrt(sum([a*a for a in x]))

	def cosine_similarity(self,x,y):
		numerator = sum(a*b for a,b in zip(x,y))
		denominator = square_rooted(x)*square_rooted(y)
		return numerator/float(denominator)

if __name__ == "__main__":
	m = MobileBERT()
	m.get_summary()
	avg = 0
	last = ""
	for x in range(0,9):
		sTime = time.time()
		last = m.run("what year was the declaration of independence signed","In fact, independence was formally declared on July 2, 1776, a date that John Adams believed would be “the most memorable epocha in the history of America.” On July 4, 1776, Congress approved the final text of the Declaration. It wasn't signed until August 2, 1776.")
		avg += (time.time()-sTime)
	print(str(avg/10)," seconds")
	print(last)
