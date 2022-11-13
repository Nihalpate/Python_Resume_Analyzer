# if you are using .pdf file as imput file please uncoment part-1

from tika import parser
from datetime import datetime
import string
import logging
logging.captureWarnings(True)

import numpy as np
import pandas as pd


from collections import Counter
from collections import defaultdict

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models
from math import log

from scipy.stats import entropy

import networkx as nx
from pyvis.network import Network

from wordcloud import WordCloud
import matplotlib.pyplot as plt






##############################################################################
## part-1



# def extract_text_pdf(file_name, len):
#     """
#     Extract a text from a given pdf file (full path required)
#     It returns a list with single words and a list of paragraphs
#     """

#     rawText = parser.from_file(file_name)

#     text_parag = rawText['content'].splitlines()
#     raw_parag_lst = []
#     for parag in text_parag:
#         if parag != '' and parag != ' ':
#             para_clean = txt_clean(parag, len)
#             raw_parag_lst. append(para_clean)
#     return raw_parag_lst


# def txt_clean(words_str, min_len):
#     """
#     Performs a first cleaning to a list of words.

#     :param words_str: string of words
#     :type: string
#     :param min_len: minimum length of a word to be acceptable
#     :type: integer
#     :return: clean_words string of clean words
#     :type: string

#     """
#     clean_words = ''
#     for word in words_str.split(' '):
#         word_l = word.lower()
#         if len(word_l) > min_len:
#             word_l1 = word_l.translate(str.maketrans('', '', string.punctuation))
#             clean_words = clean_words + ' ' + word_l1.strip()

#     return clean_words


# '''
# ----------- Main
# '''

# print ('\n-Starting the pdf to txt conversion process-\n')
# start_time = datetime.now()
# print('--- starting time: {}'.format(start_time))
# file_name_in = 'Resume-Carlo-Lipizzi_'

# min_word_len = 2
# # extracting text from all .pdf files in the specified folder path
# parag_in_lst = extract_text_pdf(file_name_in + '.pdf', min_word_len)

# with open(file_name_in + '.txt', "w") as outfile:
#     outfile.write("\n".join(parag_in_lst))

# print ('\n--- this is the end of the process ---\n')

############################################################################

############################################################################
# defining the functions 

# creates a list of n-grams. Individual words are joined together by "_"
def ngram(text,grams):  
    n_grams_list = []
    count = 0
    for token in text[:len(text)-grams+1]:  
       n_grams_list.append(text[count]+'_'+text[count+grams-1])
       count=count+1  
    return n_grams_list


# creates a list of the "num" most common elements/strings in an input list
def most_common(lst, num):
    data = Counter(lst)
    common = data.most_common(num)
    top_comm = []
    for i in range (0, num):
        top_comm.append (common [i][0])
    return top_comm


def chunk_replacement(chunk_list, text):
    for chunk in chunk_list:
        text = text.replace(chunk, chunk.replace(' ', '_'))
    return text

# extracting the topics using LDA
def topic_modeling(words, num_of_topics, file_name, algorithm='lda', file_out=True):
    """
    topic_modeling: Topic Modeling is a technique to extract the hidden topics from large volumes of text.
    Latent Dirichlet Allocation(LDA) is a topic modelling technique. In LDA with gensim is employed to create a
    dictionary from the data and then to convert to bag-of-words corpus.
    pLDAvis is designed to help users interpret the topics in a topic model that has been fit to a corpus of text data.
    The package extracts information from a fitted LDA topic model to inform an interactive web-based visualization.


    :param words: List of words to be passed
    :param num_of_topics: Define the number of topics
    :param algorithm: 'lda' - the topic modelling technique
    :param file_out: If 'TRUE' generate an html for the LDA
    :return: list of topics
    """
    tokens = [x.split() for x in words]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]

    lda = LdaModel(corpus = corpus, num_topics = num_of_topics, id2word = dictionary)
    topic_lst = []
    print ('\nthe following are the top', num_topics, 'topics:')
    for i in range (0, len(lda.show_topics(num_of_topics))):
        topic_lst.append(lda.show_topics(num_of_topics)[i][1])
        print(lda.show_topics(num_of_topics)[i][1], '\n')

    if file_out == True:
        lda_display = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(lda_display, 'LDA_' + file_name + '.html')
    return topic_lst

# creates a matrix of co-occurrence
def co_occurrence(in_text, vocab, window_size, cooccurrence_min):
    d = defaultdict(int)
    for i in range(len(in_text)):
        token = in_text[i]
        next_token = in_text[i+1 : i+1+window_size]
        for t in next_token:
            key = tuple( sorted([t, token]) )
            d[key] += 1

    # formulate the dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        if value <= cooccurrence_min:
            value = 0
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df








############################################################################
# resume analysis code 
############################################################################

import nltk 


# importing row files
input_file_name = 'resume.txt'
text_file = open(input_file_name)
stopwords_file = open('stopwords_en.txt','r', encoding='utf8')


word_tokens = []
stopwords = []
cleaned_words = []
vocabulary = []


# defining parameters
minimum_exceptable_length_of_word = 0
min_len = minimum_exceptable_length_of_word
num_topics = 5
# setting the number of elements to be similar
n_sim_elems = 10
# setting the "n" for the n-grams generated
n_value = 2
# setting the window of separation between words for the network creation
word_window = 1
# setting the minimum number of pair co-occurrence for the network creation
min_pair_coocc = 30


# populating the stopwords file
for line in stopwords_file:
    stopwords.append(line.strip()) 


# extrecting every individual words from text file 
for line in text_file:
    tokens = nltk.word_tokenize(line.strip())
    for individual_word in tokens:
        word_tokens.append(individual_word)
 
    
# cleaning the words
for i in word_tokens:
    if i not in stopwords:
        if len(i) > min_len:
            cleaned_words.append(i)
            if i not in vocabulary:
                vocabulary.append(i)


# generating biagrams and trigrams  

biagrams = ngram(cleaned_words,2)
trigrams = list(nltk.trigrams(cleaned_words)) 


# Extracting topics out of the text

all_words_string = ' '.join(cleaned_words)
ngrams_list = ngram(cleaned_words, n_value)
top_ngrams = most_common(ngrams_list, n_sim_elems)
text_chunked = chunk_replacement(top_ngrams, all_words_string)
text_chunked_lst = list(text_chunked.split(' '))
topic_list = topic_modeling (text_chunked_lst, num_topics, input_file_name)






# printing the results     
print()    
print("*** Statistical Analysis My resume of ***")
print()
print("Total number of words in the file =",len(word_tokens))
# print()
# print("Total stopwords =",len(stopwords))
print()
print("Total number of words after cleanning =",len(cleaned_words))
print()
print("Top 10 biagrams in instructor's resume")
#print(dict(Counter(biagrams)))
print([ k for k in {k: v for k, v in sorted(dict(Counter(biagrams)).items(), key=lambda item: item[1],reverse=True)}][:10])
print()
print("Top 10 trigrams in instructor's resume")
print([ k for k in {k: v for k, v in sorted(dict(Counter(trigrams)).items(), key=lambda item: item[1],reverse=True)}][:10])
# general statistics on the input text
tot_num_words = len (text_chunked_lst)
unique_words_num = len (vocabulary)
# calculating the entropy in the text
words_counter = Counter(text_chunked_lst)
word_freq_lst = list(words_counter.values())
entropy_val = entropy(word_freq_lst, base = 10)


print()    
print("*** Statistical Analysis My resume of ***")
print()
print("Total number of words in the file =",len(word_tokens))
# print()
# print("Total stopwords =",len(stopwords))
print("Total number of words after removing stopwords =",len(cleaned_words))
print ('total number of words:', tot_num_words)
print ('total number of unique words:', unique_words_num)
print ('total entropy in the text:', log(entropy_val, 10),'\n','     (entropy is a measure of information rate)')
print ('ratio of total unique words and total words in the resume =',len(word_tokens)/len (vocabulary))



# creating the statistics table and wrting to a csv file
stat_dic = {'top_ngrams':top_ngrams, 'top_topics':topic_list, 'num_words':str(tot_num_words), 'unique_words':str(unique_words_num), 'entropy':str(entropy_val)}
stat_name = 'stats_' + input_file_name

with open(stat_name + '.csv', 'w') as f:
    for key in stat_dic.keys():
        f.write("%s, %s\n" % (key, stat_dic[key]))
        
        
# generating the co-occurrence matrix and extracting a graph from the resulting adjacency matrix
adj_matrix = co_occurrence(text_chunked_lst, vocabulary, word_window, min_pair_coocc)
G = nx.from_pandas_adjacency(adj_matrix)
# visualizing the network
G_viz = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
G_viz.from_nx(G)
G_viz.show_buttons(filter_=['physics'])
graph_name = 'graph_' + input_file_name
G_viz.show(graph_name + '.html')

graph_file = graph_name + '.gml'

nx.write_gml(G, graph_file)

all_words_string = ' '.join(cleaned_words)


# Defining the wordcloud parameters
wc = WordCloud(background_color="white", max_words=2000)

# Generate word cloud
wc.generate(all_words_string)

# Store to file
wc.to_file(input_file_name+'-cloud.png')

# Show the cloud
plt.imshow(wc)
plt.axis('off')
plt.show()
