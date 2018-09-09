import gensim.models.keyedvectors as word2vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import os

model = word2vec.KeyedVectors.load('../model/word2vec_skipgram.model')
# model = word2vec.KeyedVectors.load('../model/fasttext_gensim.model')

pathfile = './words'
with open(pathfile, 'r') as f:
    words = f.readlines()
    words = [word.strip().decode('utf-8') for word in words]

words_np = []
words_label = []

for word in model.vocab.keys():
    if word in words:
        words_np.append(model[word])
        words_label.append(word)

pca = PCA(n_components=2)
pca.fit(words_np)
reduced = pca.transform(words_np)


def visualize():
    fig, ax = plt.subplots()

    for index, vec in enumerate(reduced):
        x, y = vec[0], vec[1]

        ax.scatter(x, y)
        ax.annotate(words_label[index], xy=(x, y))

    plt.show()
    return


if __name__ == '__main__':
    visualize()
