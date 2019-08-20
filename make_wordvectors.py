# coding: utf-8
#!/usr/bin/python3
import nltk
import os
import codecs
import argparse
import numpy as np

# arguments setting 
parser = argparse.ArgumentParser()
parser.add_argument('--lcode', help='ISO 639-1 code of target language. See `lcodes.txt`.')
parser.add_argument('--vector_size', type=int, default=300, help='the size of a word vector')
parser.add_argument('--window_size', type=int, default=10, help='the maximum distance between the current and predicted word within a sentence.')
parser.add_argument('--vocab_size', type=int, default=5000000, help='the maximum vocabulary size')
parser.add_argument('--num_negative', type=int, default=20, help='the int for negative specifies how many “noise words” should be drawn')
args = parser.parse_args()

lcode = args.lcode
vector_size = args.vector_size
window_size = args.window_size
vocab_size = args.vocab_size
num_negative = args.num_negative

def get_min_count(sents):
    '''
    Args:
      sents: A list of lists. E.g., [["I", "am", "a", "boy", "."], ["You", "are", "a", "girl", "."]]
     
    Returns:
      min_count: A uint. Should be set as the parameter value of word2vec `min_count`.   
    '''
    global vocab_size
    from itertools import chain
     
    fdist = nltk.FreqDist(chain.from_iterable(sents))
    min_count = fdist.most_common(vocab_size)[-1][1] # the count of the the top-kth word
    
    return min_count

def make_wordvectors():
    global lcode
    from gensim.models import Word2Vec  
    from time import time
    import multiprocessing
   
    import logging  # Setting up the loggings to monitor gensim
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


    cores = multiprocessing.cpu_count()

    t = time()
    
    sents = []
    with codecs.open('data/corpus_cleaned.txt', 'r', 'utf-8') as fin:
        while 1:
            line = fin.readline()
            if not line: break
             
            words = line.split()
            sents.append(words)

    print('Time to create sentences: {} mins'.format(round((time() - t) / 60, 2))) 

    min_count = get_min_count(sents)

    model = Word2Vec(size=vector_size, 
            min_count=min_count,
            negative=num_negative, 
            window=window_size,
            workers=cores-4)
    
    t = time()

    model.build_vocab(sents, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()

    model.train(sents, total_examples=model.corpus_count, epochs=30, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    model.init_sims(replace=True)
    
    model.save('models/spearfishing_3.bin')
    
if __name__ == "__main__":
    make_wordvectors()
    
    print("Done")
