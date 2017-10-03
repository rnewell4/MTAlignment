import numpy as np
import sys
from collections import defaultdict

def ibm(source, target):

    #Prob Source (French) aligns to Target (English)
    #Stored as tprob['target'] = {[source, prob],..}
    sentences = []
    svocab = []
    counter = 0
    for s,t in zip(source,target):

        #save as english, french
        sentences.append((t.lower(),s.lower()))
        for word in s.split():
            if word.lower() not in svocab:
                svocab.append(word.lower())
        counter += 1
        if counter % 10000 == 0:
            sys.stderr.write('.')
            break

    tprob = defaultdict(lambda: (1./len(svocab)))
    
    #print tprob
    converged = False
    firstiter = True

    itr = 0
    while itr < 10:
        sys.stderr.write(str(itr) + " ")
        counts = defaultdict(float)
        total = defaultdict(float)
        stotal = defaultdict(float)

        for sentence in sentences:

            #Compute Normalization            
            for word in sentence[0].split():
                stotal[word] = 0.
                for sword in sentence[1].split():
                    stotal[word] += tprob[(word,sword)]

            for word in sentence[0].split():
                for sword in sentence[1].split():
                    counts[(word,sword)] += tprob[(word,sword)] / stotal[word]
                    total[sword] += tprob[(word,sword)] / stotal[word]
            
            
        #estimate counts
        
        for (word,sword) in counts.keys():
            tprob[(word,sword)] = counts[(word,sword)]/total[sword]

        itr += 1
    #irint tprob
    #Write Alignments
    for sentence in sentences:
        counter = 0
        output = ""
        for x in range(len(sentence[0].split())):
            wprob = []
            for sword in sentence[1].split():
                #print sentence[0].split()[x], sword
                #print tprob[(sentence[0].split()[x],sword)]
                wprob.append(tprob[(sentence[0].split()[x],sword)])
            #print wprob
            best = wprob.index(np.max(np.array(wprob)))
            output += str(best) + "-" + str(x) + " "
        print output.strip()
        counter += 1
        

if __name__ == "__main__":

    source = open(sys.argv[1],'r')
    target = open(sys.argv[2],'r')

    ibm(source,target)