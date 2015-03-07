import time
now = time.time()

from math import log
import codecs,re,nltk
from textblob import TextBlob as tb
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem.porter import *
import numpy as np
from itertools import groupby
from pprint import pprint

def tf(word, doc):
    ''' Calculate term frequency '''
    doc = doc.lower().split()
    try:
        return round(doc.count(word.lower()) / float(len(doc)),3)
    except ZeroDivisionError:
        return round(doc.count(word.lower()) / 1.0)

def n_containing(word, corpus_values):
    ''' Count # words in each document '''
    return sum(1 for doc in corpus_values if word in doc)

def idf(word, corpus_values):
    num_texts_with_word = len([True for text in corpus_values if word.lower()
                              in text.lower().split()])
    try:
        return log(float(len(corpus_values)) / (1 + n_containing(word,corpus_values)))
        #return 1.0 + log(float(len(corpus_values)) / num_texts_with_word)
    except ZeroDivisionError:
        return 1.0

def tf_idf(word, doc, corpus_values):
    return round(tf(word, doc) * idf(word, corpus_values),5)

tag_dict = {}
tag_dict_values = []
def input_tags(infile):
    with codecs.open(infile,'r',encoding='utf-8',errors='ignore') as f:
        x = []
        for line in f:
            pid = line.split('\t')[1]
            tag = line.split('\t')[2]
            try:
                x = tb(str(pid)) + '\t' + tb(str(tag))
            except:
                x = tb(str(pid))
            try:
                both = {pid: x.split('\t')[1]}
            except:
                continue
            tag_dict.update(both)
            tag_dict_values.append(tag)

ww = []
def process_tags():
    stemmer = SnowballStemmer("english")
    stemmer = PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()
    for upc,doc in tag_dict.items():
        words = [ w.strip() for w in doc.split(';')]
        unique_words = np.unique(words).tolist()
        scores = {word: tf_idf(word, tag_dict[upc], tag_dict_values) for word in unique_words}
        #sorted_words = sorted(scores.items(), key=lambda x: x[1]) ###produces LONGER tags & MORE author/character names
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True) ###produces SHORTER tags & LESS author/character names
        #pprint(sorted_words)
        for tag,score in sorted_words[:10]:
            if re.search('[a-zA-Z]', tag):
                if tag.lower() not in ['\'s','book','books','booklet','s','novel','ebook','paperback']:
                    if tag.lower() not in stopwords.words('english'):
                        tag_lemma = lemmatizer.lemmatize(tag.lower())
                        words_tuple = upc,tag_lemma
                        ww.append(words_tuple)

def output_tags(outfile):
    with codecs.open(outfile,'w',encoding='utf-8',errors='ignore') as f:
        for key, group in groupby(ww,lambda x: x[0]):
            listofThings = ','.join(['%s' %thing[1] for thing in group])
            tags = [t for t in listofThings.split(',')]
            #print tags
            unique_tags = np.unique(tags)
            str1 = ','.join(unique_tags).strip('\r\n')
            print key + ',' + str1
            print >> f,key + ',' + str1

def main(infile,outfile):
    input_tags(infile)
    process_tags()
    output_tags(outfile)
    print ''
    print '-'*40
    print "It took %0.02d secs to run this program" % (time.time() - now)
    t=time.localtime()
    print 'Date:',time.strftime("%a, %d %b %Y",t)
    print 'Time:',time.strftime("%H:%M:%S",t)
    print '-'*40

if __name__ == '__main__':
    infile = '../foldername/LibraryThingTest.txt'
    outfile = '../foldername/test_tag_out.txt'
    main(infile,outfile)
