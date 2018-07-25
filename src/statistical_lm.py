from srilm import readLM, initLM, getNgramProb, getIndexForWord, howManyNgrams

class SRILM_lm_loader(object):
	"""TODO"""
	def __init__(self, model_path, order=2): 
		self.lm_path = model_path
		self.order=order
		self.history=[]
		self.history_len = order-1
                self.lm = initLM(order)
                readLM(self.lm, self.lm_path)
                self.vocab_size = howManyNgrams(self.lm, 1)
	
	def predict_next_seq(self, sequence):
		"""retrieve the probability of a sequence from the language model
		 based on the current history"""
		prefix = "%s " % ' '.join(self.history)
		#~ logging.debug(u'LM over chars prefix: {}'.format(prefix))
		
		order = len(self.history) + 1

		ret = getNgramProb(self.lm, prefix + ("</s>" if w == utils.EOS_ID else str(w)), order)
		#~ logging.debug(u'LM over chars distribution: {}'.format(ret))
		return ret
	
	def predict_next_morph(self, morphemes, eow=0):
		"""Score the set of target MORPHEMES with the n-gram language model given the current history of MORPHEMES.
		
		Args:
		words (list): Set of morphemes to score
		Returns:
		dict. Language model scores for the words in ``words``
		"""
		prefix = "%s " % ' '.join(self.history)
		order = len(self.history) + 1
			
		if eow==1:
			# Score for the end of word symbol:
			# logP(second-last-morf last-morf morf </s>) = logP(second-last-morf last-morf morf) + logP(last-morf morf </s>)
			#prefix_eos = "%s " % ' '.join(self.history[1:]) if len(self.history) > self.history_len else "%s " % ' '.join(self.history)
			prefix_eos = "%s " % ' '.join(self.history[1:]) if len(self.history) == self.history_len else "%s " % ' '.join(self.history)
			#prefix_eos = "%s " % ' '.join(self.history[1:])
			#order_eos = order+1
			order_eos = order if len(self.history) == self.history_len else order+1
			#order_eos = order
			#~ logging.debug(u"prefix: {} w: {} order: {} score {}".format(prefix + str(morphemes[0]),str(morphemes[0]),order,getNgramProb(self.lm, prefix + str(morphemes[0]), order)))
			
			#~ logging.debug(u"prefix_eos: {} w: {} order: {} score {}".format(prefix_eos + str(morphemes[0]) + " </s>", " </s>",order_eos, getNgramProb(self.lm, prefix_eos + str(morphemes[0]) + " </s>", order_eos)))
			
			prob = {w: (getNgramProb(self.lm, prefix + str(w), order) + getNgramProb(self.lm, prefix_eos + str(w) + " </s>", order_eos)) for w in morphemes}
					
		else:
			#~ logging.debug(u"prefix: {} w: {} score {}".format(prefix,str(morphemes[0]),getNgramProb(self.lm, prefix + str(morphemes[0]), order)))
			# Score for the segmentation boundary symbol:
			prob = {w: getNgramProb(self.lm, prefix + str(w), order) * scaling_factor for w in morphemes}
		
		return prob
	
	def consume(self):
		"""add predicted item to vocabulary"""
		if len(self.history) >= self.history_len:
			self.history = self.history[1:]
		self.history.append(str(word))
	
	def get_state(self):
		"""Returns the current n-gram history """
		return self.history
	
	def set_state(self, state):
		"""Sets the current n-gram history """
		self.history = state

if __name__=="__main__":
    import sys
    sri_lm = SRILM_lm_loader(sys.argv[1],order=7)
