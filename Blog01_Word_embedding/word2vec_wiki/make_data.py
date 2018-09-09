from pyvi.ViTokenizer import tokenize
import re, os, string
import pandas as pd

def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()

# list stopwords
filename = './stopwords.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_stopwords = data['stopwords']

def remove_stopword(text):
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
    text2 = ' '.join(pre_text)

    return text2

def sentence_segment(text):
    sents = re.split("([.?!])?[\n]+|[.?!] ", text)
    return sents


def word_segment(sent):
    sent = tokenize(sent.decode('utf-8'))
    return sent

path_to_corpus = '../wikipediacorpus'


f_w = open('./datatrain.txt', 'w')
for i, sub_dir in enumerate(os.listdir(path_to_corpus)):
    path_to_subdir = path_to_corpus + '/' + sub_dir
    for j, file_name in enumerate(os.listdir(path_to_subdir)):
        with open(path_to_subdir + '/' + file_name) as f_r:
            contents = f_r.read().strip().split('</doc>')
            for content in contents:
                if (len(content) < 5):
                    continue
                content = clean_text(content)
                sents = sentence_segment(content)
                for sent in sents:
                    if(sent != None):
                        sent = word_segment(sent)
                        sent = remove_stopword(normalize_text(sent))
                        if(len(sent.split()) > 1):
                            f_w.write(sent.encode('utf-8') + '\n')
            print "Done ", i + 1, ':', j + 1

f_w.close()