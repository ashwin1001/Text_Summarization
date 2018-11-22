import os
import nltk  # importing nltk package for preprocessing steps
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from itertools import combinations
import re
import math
import networkx as nx

G = nx.DiGraph()
Y=nx.Graph()
from operator import itemgetter

sentences = {}              #for storing the actual sentences with corresponding number
tokenize_words = {}         #storing the sentences in form of tokenized words
tf_idf_vector = {}          #storign the tf_idf vector for every sentence
idf = {}                    #storing the idf for rvery word
page_rank={}                #for storing pagerank
last_page_rank={}           #for storing last pagerank
in_nodes={}                 #for storing incoming nodes
out_degree={}               #for storing out degree of nodes
length=0

#function for calculating the tf_id vector of all the sentences
#It takes the dictionary of tokenize_words and then form the vector of tfid values for every 
#It also calculates the idf for every word and stores it in the idf vector
def calculate_tf_idf():
    global idf
    for key,val in tokenize_words.iteritems():
	weight = []
        for elements in val:
	    
            tf = 0
            df = 0
            for k,v in tokenize_words.iteritems():
                if  elements in v:
                    df+=1

            tf = val.count(elements)
            value = 0
            idf[elements] = math.log10(len(tokenize_words)/df)
            if df == 0:
                print 'df==0'
                print str(k) + '----' + str(df)

            if df != 0:
                value = tf * math.log10(len(tokenize_words) / df)
            weight.append(value)

        tf_idf_vector[key] = []
        tf_idf_vector[key] = weight


i = 0

#Function for preprocessing data
#It takes the data form the topic files and then preprocess them and stores it in the respective dictionary
def preprocess_data(data):
    global i,length
    # setting stopword removal object for english
    stop_word_set = set(stopwords.words('english'))
    stop_word_set.update(['.', ',', '"', "'", '\xe2', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

    data = re.sub('<[^<]+>', "", data)
    sent = sent_tokenize(data)
    for k in sent:
        k = k.strip()
        sent = k
        sent = re.sub('[\n]', '', sent)

        sentences[i] = sent
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(k)
        tokes = [k for k in tokens if k not in stop_word_set]
        tokenize_words[i] = tokes
        #calculate_tf_idf(i, tokens)
        i += 1
        # print k
    length=len(tokenize_words)

#This function fetches data from the files and then calls the Preprocessing function for further processing
#It is called in main function
def fetch_data(path):
    for filename in os.listdir(path):
        filepath = path + '/' + filename
        # print filepath
        f = open(filepath, 'r')
        content = f.read()
        content = content.strip()

        content = content.split('<TEXT>')
        content = content[1].split('</TEXT>')
        preprocess_data(content[0])
        f.close()

#This function is called by the form graph function for calculating the tfidf scores between two sentences
#It takes to sentences as input and calculates their tf_idf scores
def tf_idf_calculation(k,t):
    s1 = tokenize_words[k]
    s2 = tokenize_words[t]
    common = list(set(s1).intersection(set(s2)))
    numerator = 0
    for l in common:
        if idf[l] != 0:
            numerator += tokenize_words[k].count(l) * tokenize_words[t].count(l) * math.log10(
                length / idf[l]) * math.log10(length / idf[l])
    sum1 = 0
    sum2 = 0
    for z in tf_idf_vector[k]:
        sum1 += z * z
    for p in tf_idf_vector[t]:
        sum2 += p * p
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if denominator==0:
        weight=0
    else:
        weight=numerator/denominator
    
    return weight

#Graph used for degreee= centrality
#called in main function
def Form_Graph1():
    global Y
    Y.add_nodes_from(sentences.keys())
    nodePairs = list(combinations(sentences.keys(), 2))
    for pair in nodePairs:

        node1 = pair[0]
        node2 = pair[1]
        s1 = tokenize_words[node1]
        s2 = tokenize_words[node2]
        simval = tf_idf_calculation(node1,node2)
        if simval > 0.1:
            Y.add_edge(node1, node2, weight=simval)


#This function forms the graph from the sentences
#it is called in the main function
def Form_Graph():
    for k in range(len(sentences) - 1):
        for t in range(len(sentences) - 1):
            
            if k != t:
                weight1=tf_idf_calculation(k,t)

                if weight1 < 0.2:
                    G.add_edge(k, t, weight=0)
                    #G.add_edge(t,k,weight=0)
                else:
                    G.add_edge(k, t, weight=weight1)
                    #G.add_edge(t, k, weight=weight1)

#this function calculates the pagerank of the graph and returns the score in the main function
def calculate_Pagerank():
    global in_nodes
    nodes1=G.nodes()
    for node in nodes1:
        page_rank[node]=0.15
        last_page_rank[node]=0.15
    i=1

    for nodes in nodes1:

        inedges = G.in_edges(nodes)

        if len(inedges) == 0:
            continue

        in_nodes[nodes] = set()
        for k in inedges:
            in_nodes[nodes].add(k[0])

        pagerank_factor = 0
        for l in in_nodes[nodes]:
            out_degree[l]=G.out_degree(l)
            pagerank_factor += page_rank[l] / (out_degree[l])

        last_page_rank[nodes] = 0.15 + 0.85 * pagerank_factor

    while i <100:
        count=0
        for nodes in nodes1:

            if len(in_nodes[nodes])==0:
                continue


            pagerank_factor=0
            for l in in_nodes[nodes]:
                pagerank_factor+=page_rank[l]/(out_degree[nodes])

            page_rank[nodes]=0.15+0.85*pagerank_factor
            if abs(last_page_rank[nodes]-page_rank[nodes])>0.001:
                count=1
        i+=1
        
        if count==0:
            break
        for key,val in page_rank.iteritems():
            last_page_rank[key]=val

    scores = sorted(page_rank.items(), key=itemgetter(1), reverse=True)
    return scores


#This function is used for fetching sentences based on degree centrality
def get_keysentences():
    impsentence=[]
    word_count=0
    while True:
        if word_count>245:
            break
        #calculating the degree centrality of the graph
        calculated_degree_centrality = nx.degree_centrality(Y)
	    #sorting the data on the degreee centrality scores based on the values
        keysentences = sorted(calculated_degree_centrality.items(), key=calculated_degree_centrality.get, reverse=True)

        #Remove nodes highly similar to the the highest similarity sentence
        element=keysentences[0]
        impsentence.append(sentences[element[0]])
        node = keysentences[0]
        word_count+=len(word_tokenize(sentences[node[0]]))
        for n in Y.neighbors(node[0]):
            simval = tf_idf_calculation(node[0],n)
            if (simval>0.7):
                Y.remove_node(n)
        Y.remove_node(node[0])
    return impsentence

def print_dg(sent):
    file = open('summary/summary10.1DG.txt', 'w')
    for element in sent:
        file.write(str(sent) + '\n')
    file.close()

def print_tr(scores):
    word_number = 0
    file = open('summary/summary50.1TR.txt', 'w')
    for k, v in scores:
        if word_number > 250:
            break
        word_number += len(word_tokenize(sentences[k]))
        file.write(str(sentences[k]) + '\n')

    file.close()

#main function whch calls the other function for different problems
if __name__ == "__main__":
    path = 'data/Topic5'        #fetching the data path
    fetch_data(path)            #fetching the data path
    calculate_tf_idf()		#for calculating the tf_id scores of the sentences

    Form_Graph()                #forming the graph from the fetched sentences
    scores=calculate_Pagerank() #calculating the pagerank

    #Form_Graph1()
    print_tr(scores)

    #sent=get_keysentences()
    #print_dg(sent)
    
    '''#printing the sentences into the file
    
'''

