#!/usr/bin/env python
import optparse
import sys
import numpy as np
import random
import itertools
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
if opts.num_sents < len(bitext) and opts.num_sents > 0:
  bitext = bitext[:opts.num_sents]

vocab_e = set()
vocab_f = set()
for (f,e) in bitext:
  vocab_f.update(f)
  vocab_e.update(e)

trans_prob = defaultdict(lambda: 1./(len(vocab_f)*len(vocab_e)))


align_prob = defaultdict(lambda: 0.0)
for (f,e) in bitext:
  for (i,j) in itertools.product(range(len(e)),range(len(f))):
    align_prob[(i,j,len(e),len(f))] += 1./(len(e) + 1) 


#print align_prob.keys()

for i in range(50):
  total = defaultdict(float)
  fe_count = defaultdict(float)
  align_total = defaultdict(float)
  align_count = defaultdict(float)
  for (n, (f,e)) in enumerate(bitext):

    ##Calculate normalizing Constant
    s_total = defaultdict(float)
    for (i, e_word) in enumerate(e):
      for (j,f_word) in enumerate(f):
        s_total[e_word] += trans_prob[(e_word, f_word)]*align_prob[(i,j, len(e), len(f))]

    ###Expect###
    for (i,e_word) in enumerate(e):
      for (j,f_word) in enumerate(f):
        weight = trans_prob[(e_word, f_word)]*align_prob[(i,j,len(e),len(f))]/s_total[e_word]
        fe_count[(e_word, f_word)] += weight
        total[f_word] += weight
        align_count[(i,j,len(e),len(f))] += weight
        align_total[(j,len(e),len(f))] += weight

  ###Maximize####
  for (e_word,f_word) in fe_count:
      trans_prob[(e_word, f_word)] = fe_count[(e_word, f_word)]/total[f_word]
  for (i,j, l_e, l_f) in align_count:
    align_prob[(i,j,l_e,l_f)] = align_count[(i,j,l_e,l_f)]/align_total[(j,l_e,l_f)]


#print align_prob.keys()
for (f,e) in bitext:
  output = ""
  for index,eword in enumerate(e):
    prob = np.multiply([trans_prob[(eword, fword)] for fword in f],
                       [align_prob[(index, j, len(e),len(f))] for j in range(len(f))])
    #prob = [trans_prob[(eword, fword)] for fword in f]
    output += "%i-%i " %(np.argmax(prob), index)
  print output 
