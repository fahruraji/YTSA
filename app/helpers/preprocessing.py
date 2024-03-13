from app.model import Negasi, Stopword, Informal

## 1. CLEANSING
import string
import re

def casefolding(text):    
  text = str(text).lower() # Menyeragamkan kapitalisasi (casefolding)
  text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()) # Menghapus mention, karakter selain alfanumerik dan link web
  text = re.sub(r'\$\w*', '', text) # Menghapus variabel php yang dimulai dengan tanda dollar
  text = re.sub(r'^RT[\s]+', '', text) # Menghapus Retweet
  text = re.sub(r'#', '', text) # Menghapus hashtag
  text = re.sub(r'[0-9]+', '', text)  # Menghapus angka
  text = text.strip()  # Menghapus kelebihan spasi
  return text

## 2. TOKENIZING
import nltk
from nltk.tokenize import word_tokenize

def tokenizing(text):
  tokens = nltk.word_tokenize(text)
  return tokens

## 3. NORMALIZING
def normalizing(tokens):    
    content = [Informal.query.filter_by(bentuk_informal=token).first().bentuk_formal if (informal:=Informal.query.filter_by(bentuk_informal=token).first()) else token for token in tokens]
    result = ' '.join(content)
    return result

## 4. STEMMING
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
  text = stemmer.stem(text)
  return text

## 5. NEGATION HANDLING
negasi = [word for word, in Negasi.query.with_entities(Negasi.kata_negasi).all()]
def negation_handling(text):
    # Membuat pola regex untuk menemukan kata-kata negasi
    negation_pattern = re.compile(r'\b(?:' + '|'.join(negasi) + r')\b\s+(\w+)')
    # Mengganti kata dengan menambahkan "NOT_"
    result = negation_pattern.sub(lambda match: 'NOT_' + match.group(1), text)
    return result

## 6. FILTERING
stopword = [word for word, in Stopword.query.with_entities(Stopword.stop_word).all()]
def filtering(text):
  text = text.replace("NOT_", "") # Menghapus penanda negasi
  tokens = nltk.word_tokenize(text)

  content = []
  for token in tokens:
    if token not in stopword:
      content.append(token)
  result = ' '.join(content)
  return result

# ## 6. POS TAGGING
# import stanza
# import logging
# logging.getLogger('stanza').setLevel(logging.WARNING)
# nlp = stanza.Pipeline('id', processors='tokenize,pos,lemma,mwt')
# def pos_tagging(text):
#     doc = nlp(text)
#     pos_tokens = [f'{word.text} ({word.pos})' for sentence in doc.sentences for word in sentence.words]
#     return ' '.join(pos_tokens)

# ## 7. LEMMATIZING
# def lemmatizing(text):
#     doc = nlp(text)
#     lemmatized_tokens = [word.lemma for sentence in doc.sentences for word in sentence.words]
#     return ' '.join(lemmatized_tokens)

def preprocessing(text):
    text = casefolding(text)
    text = tokenizing(text)
    text = normalizing(text)
    text = stemming(text)
    return text

def preprocessing_1(text):
    text = casefolding(text)
    text = tokenizing(text)
    text = normalizing(text)
    return text

def preprocessing_2(text):
    text = stemming(text)
    return text