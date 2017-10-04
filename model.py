"""
Alignment Models
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict, Counter
import sys
import numpy as np
from tqdm import tqdm

from alphabet import Alphabet

class AlignmentModel(object, metaclass=ABCMeta):
	"""docstring for AlignmentModel"""

	@abstractmethod
	def train(self, bitext):
		pass

	@abstractmethod
	def align(self, french, english):
		pass

	@property
	@abstractmethod
	def name(self):
		pass

class DiceCoefficient(AlignmentModel):
	"""docstring for DiceCoefficient"""
	def __init__(self, threshold):
		super().__init__()
		self.threshold = threshold
		
	def train(self, bitext):
		print("Training with Dice's coefficient...", file=sys.stderr)
		f_count = Counter()
		e_count = Counter()
		fe_count = Counter()
		for n, (f, e) in enumerate(bitext):
			sf = frozenset(f)
			se = frozenset(e)
			fe_count.update(zip(sf, se))
			f_count.update(sf)
			e_count.update(se)
			if n % 500 == 0:
				sys.stderr.write(".")

		self.dice = defaultdict(int)
		for k, (f_i, e_j) in enumerate(fe_count):
			self.dice[f_i,e_j] = 2.0 * fe_count[f_i, e_j] / (f_count[f_i] + e_count[e_j])
			if k % 5000 == 0:
				sys.stderr.write(".")
		sys.stderr.write("\n")

	def align(self, french, english):
		return [
			(i, j)
			for i, f_i in enumerate(french)
			for j, e_j in enumerate(english)
			if self.dice[f_i, e_j] >= self.threshold
		]
		

	@property
	def name(self):
		return 'dice'


class IBMModel1(AlignmentModel):
	"""docstring for IBMModel1"""
	def __init__(self, epochs=10):
		super().__init__()
		self.epochs = epochs

	def train(self, bitextGen):
		self.frenchAlphabet = Alphabet.from_iterable(word for frSent, enSent in bitextGen(desc='French Alphabet') for word in frSent)
		self.englishAlphabet = Alphabet.from_iterable(word for frSent, enSent in bitextGen(desc='English Alphabet') for word in enSent)
		self.frenchAlphabet.freeze()
		self.englishAlphabet.freeze()
		vF = len(self.frenchAlphabet)
		vE = len(self.englishAlphabet)
		tOfEGivenF = np.ones((vE, vF)) / vF
		for ep in tqdm(range(self.epochs), desc='Epoch'):
			countOfEGivenF = np.zeros((vE, vF))
			totalOfF = np.zeros(vF)
			for frSent, enSent in bitextGen('Training'):
				# Compute Normalization stuff
				frMask = self.frenchAlphabet.map(frSent)

				enMask = self.englishAlphabet.map(enSent)

				# total probability of each english word being translated from the french ones
				# has size of {len(enSent) x 1}
				sTotalOfE = np.sum(tOfEGivenF[np.ix_(enMask, frMask)], axis=1, keepdims=True)

				# calculate counts
				
				delta = tOfEGivenF[np.ix_(enMask, frMask)] / sTotalOfE
				countOfEGivenF[np.ix_(enMask, frMask)] += delta
				totalOfF[frMask] += np.sum(delta, axis=0)

			# estimate probabilities
			tOfEGivenF = countOfEGivenF / totalOfF

		self.tOfEGivenF = tOfEGivenF

	def align(self, frSent, enSent):

		EGivenFSentences = self.tOfEGivenF[
			np.ix_(
				self.englishAlphabet.map(enSent),
				self.frenchAlphabet.map(frSent)
			)
		]

		alignments = np.argmax(
			EGivenFSentences,
			axis=1
		)

		assert len(alignments) == len(enSent)

		return ((i, j) for j, i in enumerate(alignments))

	@property
	def name(self):
		return 'ibm-model-1'


class AlignmentDict(dict):
	def __missing__(self, key):
		lenE, lenF = key
		value = self[key] = np.ones((lenE, lenF)) / lenE
		return value

class CountDict(dict):
	def __missing__(self, key):
		lenE, lenF = key
		value = self[key] = np.zeros(lenF)
		return value


class IBMModel2(AlignmentModel):
	"""docstring for IBMModel2"""
	def __init__(self, epochs=10):
		super().__init__()
		self.epochs = epochs

	def train(self, bitextGen):
		self.frenchAlphabet = Alphabet.from_iterable(word for frSent, enSent in bitextGen(desc='French Alphabet') for word in frSent)
		self.englishAlphabet = Alphabet.from_iterable(word for frSent, enSent in bitextGen(desc='English Alphabet') for word in enSent)
		self.frenchAlphabet.freeze()
		self.englishAlphabet.freeze()
		vF = len(self.frenchAlphabet)
		vE = len(self.englishAlphabet)
		tOfEGivenF = np.ones((vE, vF)) / vF
		aOfIJGivenLenELenF = AlignmentDict()
		for ep in tqdm(range(self.epochs), desc='Epoch'):
			countOfEGivenF = np.zeros((vE, vF))
			totalOfF = np.zeros(vF)
			countOfIGivenJ = AlignmentDict()
			totalOfJ = CountDict()
			for frSent, enSent in bitextGen('Training'):
				# Compute Normalization stuff
				lenF = len(frSent)
				frMask = self.frenchAlphabet.map(frSent)

				lenE = len(enSent)
				enMask = self.englishAlphabet.map(enSent)

				aOfIJ = aOfIJGivenLenELenF[lenE, lenF]

				# total probability of each english word being translated from the french ones
				# has size of {len(enSent) x 1}
				sTotalOfE = np.sum(
					tOfEGivenF[np.ix_(enMask, frMask)] * aOfIJ,
					axis=1,
					keepdims=True
				)
				

				# calculate counts
				
				delta = tOfEGivenF[np.ix_(enMask, frMask)] * aOfIJ / sTotalOfE
				deltaSummedOverE = np.sum(delta, axis=0)

				countOfEGivenF[np.ix_(enMask, frMask)] += delta
				totalOfF[frMask] += deltaSummedOverE

				countOfIGivenJ[lenE, lenF] += delta
				totalOfJ[lenE, lenF] += deltaSummedOverE


			# estimate probabilities
			tOfEGivenF = countOfEGivenF / totalOfF
			for lenE, lenF in aOfIJGivenLenELenF:
				aOfIJGivenLenELenF[lenE, lenF] = countOfIGivenJ[lenE, lenF] / totalOfJ[lenE, lenF]
			

		self.tOfEGivenF = tOfEGivenF
		self.aOfIJGivenLenELenF = aOfIJGivenLenELenF

	def align(self, frSent, enSent):

		EGivenFSentences = self.tOfEGivenF[
			np.ix_(
				self.englishAlphabet.map(enSent),
				self.frenchAlphabet.map(frSent)
			)
		]

		aOfIJ = self.aOfIJGivenLenELenF[len(enSent), len(frSent)]

		alignments = np.argmax(
			EGivenFSentences * aOfIJ,
			axis=1
		)

		assert len(alignments) == len(enSent)

		return ((i, j) for j, i in enumerate(alignments))

	@property
	def name(self):
		return 'ibm-model-2'

		
