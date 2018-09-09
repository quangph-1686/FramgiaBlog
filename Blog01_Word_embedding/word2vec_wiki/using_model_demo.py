# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np

model = KeyedVectors.load('../model/word2vec_skipgram.model')

for word in model.most_similar(u"quang"):
    print word[0]