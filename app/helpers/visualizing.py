import nltk
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def dist_freq(dataset, limit):
    # Tokenisasi teks pada kolom 'text' di dataframe
    tokens = []
    for text in dataset:
        tokens += nltk.word_tokenize(text)
    # Menghitung frekuensi kemunculan setiap kata dalam teks input pengguna
    input_word_freq = Counter(tokens)
    
    # Menampilkan 10 kata terbanyak muncul pada teks input pengguna
    top_words = input_word_freq.most_common(limit)
    
    return top_words

def data_to_json(data, lbl, val):
    data_json = [[lbl, val]]
    for item in data:
        label = item[0]
        value = item[1]
        data_json.append([label, value])
        
    return data_json

# Define the color function
def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(240, 100%%, %d%%)" % np.random.randint(40, 70)

def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(360, 100%%, %d%%)" % np.random.randint(40, 70)

def generate_wordcloud(text, sentimen, id, mask):    
    if sentimen == 'positif':
        color = green_color_func
    else:
        color = red_color_func
    
    folder = os.path.join('app', 'static', 'img', 'wordclouds')
    filename = folder+"\\"+sentimen+"\\"+id+".png"
    wordcloud = WordCloud(background_color='white', collocations=False, mask=mask, width=1600, height=800, color_func=color).generate(' '.join(text))
    if os.path.exists(filename):
        os.remove(filename)
    wordcloud.to_file(filename)

    # return plot_url
    
def associate_data(data):
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_items = apriori(df, min_suport=0.5, use_colnames=True)
    
    association_rules = association_rules(frequent_items, metric="confidence", min_threshold=0.7)
    
    return association_rules
