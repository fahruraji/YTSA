# import os

# ## INDOBERT
# import torch
# import joblib
# from torch import optim
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# from app import app
# from app.utils.data_utils import DocumentSentimentDataset
# from app.helpers.preprocessing import casefolding, tokenizing, filtering, normalizing, stemming, negation_handling, preprocessing

# class IndobertClassifier:
#     def __init__(self):
#         # Sentiment model setup
#         sentiment_path = os.path.join('app', 'pkl', 'indobert', 'sentiment')
#         self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_path)
#         sentiment_config = AutoConfig.from_pretrained(sentiment_path)
#         sentiment_config.num_labels = DocumentSentimentDataset.NUM_LABELS
#         self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_path, config=sentiment_config)

#         self.s2w, self.w2s = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL

#         # Intent model setup
#         intent_path = os.path.join('app', 'pkl', 'indobert', 'intent')
#         INDEX2LABEL = joblib.load(os.path.join(intent_path, 'i2w.joblib'))
#         LABEL2INDEX = joblib.load(os.path.join(intent_path, 'w2i.joblib'))
#         self.intent_tokenizer = AutoTokenizer.from_pretrained(intent_path)
#         intent_config = AutoConfig.from_pretrained(intent_path)
#         intent_config.num_labels = len(INDEX2LABEL)
#         self.intent_model = AutoModelForSequenceClassification.from_pretrained(intent_path, config=intent_config)

#         self.i2w, self.w2i = LABEL2INDEX, INDEX2LABEL

#     def _predict(self, tokenizer, model, text):
#         subwords = tokenizer.encode(text)
#         if len(subwords) > 512:
#             subwords = subwords[:512]
#         subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

#         logits = model(subwords)[0]
#         label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
#         trust = f'{F.softmax(logits, dim=-1).squeeze()[label] * 100:.2f}%'

#         return subwords, label, trust

#     def predict(self, text, predict_type='sentiment', return_type='label'):
#         def process_result(subwords, label, trust, label_mapping):
#             if return_type == 'encode':
#                 return subwords
#             elif return_type == 'label':
#                 return f'{label_mapping[label]}'
#             elif return_type == 'trust':
#                 return trust
#             elif return_type == 'label+trust':
#                 return f'{label_mapping[label]} {trust}'

#         if predict_type == 'sentiment':
#             tokenizer, model, label_mapping = self.sentiment_tokenizer, self.sentiment_model, self.w2s
#         elif predict_type == 'intent':
#             tokenizer, model, label_mapping = self.intent_tokenizer, self.intent_model, self.w2i
#         else:
#             # Handle unknown prediction type
#             return None

#         return process_result(*self._predict(tokenizer, model, text), label_mapping)



# ## INDOBERT
# import torch
# from torch import optim
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
# from app.utils.data_utils import DocumentSentimentDataset

# model_path = os.path.join('app', 'pkl', 'indobert', 'sentiment')

# indobert_tokenizer = AutoTokenizer.from_pretrained(model_path)
# indobert_config = AutoConfig.from_pretrained(model_path)
# indobert_config.num_labels = DocumentSentimentDataset.NUM_LABELS
# indobert_model = AutoModelForSequenceClassification.from_pretrained(model_path, config=indobert_config)

# w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL

# def indobert_encode(text):
#     subwords = indobert_tokenizer.encode(text)
#     if len(subwords) > 512:
#         subwords = subwords[:512]
#     subwords = torch.LongTensor(subwords).view(1, -1).to(indobert_model.device)
        
#     return subwords

# def indobert_classify(text):
#     subwords = indobert_encode(text)
#     logits = indobert_model(subwords)[0]
#     label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

#     if i2w[label] == 'negative':
#         label = 'negatif'
#     elif i2w[label] == 'neutral':
#         label = 'netral'
#     else:
#         label = 'positif'
   
#     return label
    
# def indobert_trust(text):
#     subwords = indobert_encode(text)
#     logits = indobert_model(subwords)[0]
#     label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

#     return f'{F.softmax(logits, dim=-1).squeeze()[label] * 100:.2f}%'

# ## MACHINE LEARNING ALGORITHM

# import pickle
# nbc_model = os.path.join('app', 'pkl', 'naive_bayes_model.pkl')
# svm_model = os.path.join('app', 'pkl', 'svm_bagging_model.pkl')
# vectorizer = os.path.join('app', 'pkl', 'tfidf_vectorizer.pkl')

# def label(prediksi):
#     if prediksi == 1:
#         return 'positif'
#     elif prediksi == 0:
#         return 'netral'
#     else:
#         return 'negatif'

# def vectorize(text):
#     with open(vectorizer, 'rb') as f:
#         vectorize = pickle.load(f)
    
#     vectors = vectorize.transform([text])
    
#     return vectors

# def svm_classify(text):
#     with open(svm_model, 'rb') as f:
#         clf = pickle.load(f)
        
#     vectors = vectorize(text)       
#     prediksi = clf.predict(vectors)
    
#     return label(prediksi)
    
# def nbc_classify(text):
#     with open(nbc_model, 'rb') as f:
#         clf = pickle.load(f)
    
#     vectors = vectorize(text)       
#     prediksi = clf.predict(vectors)
    
#     return label(prediksi)


##################################
##        LEXICON-BASED         ##
##################################

    
from langdetect import detect
from googletrans import Translator
translator = Translator()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def terjemahkan(word):
    try:
        return translator.translate(word, dest='en', src='id').text
    except:      
        return word


from textblob import TextBlob

def textblob_classify(text):
    blob = TextBlob(terjemahkan(text)).sentiment
    return blob.polarity
    
def vader_classify(text):

    text = terjemahkan(text)

    scores = analyzer.polarity_scores(text)
  
    if scores['compound'] > 0:
        predicted_labels = 'positive'
    elif scores['compound'] < 0:
        predicted_labels = 'negative'
    else:
        predicted_labels = 'neutral'

    return predicted_labels


###########################################
##               INDOBERT                ##
###########################################

import torch
import joblib
# import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW, Adamax, SparseAdam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

model_path = os.path.join('app', 'ml')
image_path = os.path.join('app', 'static', 'img', 'charts')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).clone().detach()
        return item

class IndoBERTFineTuner:
    def __init__(self, mode, X_train, y_train, X_test, y_test, pretrained_model='indolem/indobertweet-base-uncased', optimizer_name='AdamW', max_length=64, batch_size=8,
                 learning_rate=1e-5, epochs=5, dropout_rate=0.1, l2_reg=0.01, early_stopping_patience=3, scheduler_warmup_steps=100, gradient_clip_value=1.0):
        self.optimizer_name = optimizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.early_stopping_patience = early_stopping_patience
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.gradient_clip_value = gradient_clip_value
        self.pretrained_model = pretrained_model
        self.model_path = os.path.join(model_path, 'indobert', mode)
        self.image_path = os.path.join(image_path, 'indobert', mode)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def create_directories(self):
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)

    def load_label_info(self):
        label_info_path = os.path.join(self.model_path, 'i2w.joblib')
        if os.path.exists(label_info_path):
            label_info = joblib.load(label_info_path)
            self.label_encoder = label_info['label_encoder']
            self.INDEX2LABEL = label_info['INDEX2LABEL']
        else:
            self.label_encoder = LabelEncoder()
            self.INDEX2LABEL = {}

    def save_label_info(self):
        label_info = {
            'label_encoder': self.label_encoder,
            'INDEX2LABEL': self.INDEX2LABEL
        }
        label_info_path = os.path.join(self.model_path, 'i2w.joblib')
        joblib.dump(label_info, label_info_path)

    def decode_labels(self, encoded_labels):
        decoded_labels = [self.INDEX2LABEL[label] for label in encoded_labels]
        return decoded_labels
    
    def load_data(self):
        self.load_label_info()

        y_encoded = self.label_encoder.fit_transform(self.y_train)
        y_decoded = self.label_encoder.inverse_transform(y_encoded)

        for index, label in zip(y_encoded, y_decoded):
            self.INDEX2LABEL[index] = label

        self.save_label_info()

        X_val, X_test, y_val, y_test = train_test_split(self.X_test, self.y_test, test_size=0.5, random_state=42)

        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)

        X_train, X_val, X_test, y_train, y_val, y_test = self.X_train, X_val, X_test, y_encoded, y_val_encoded, y_test_encoded

        return X_train, y_train, X_val, y_val, X_test, y_test

    def prepare_dataloaders(self, tokenizer, X_train, y_train, X_val, y_val, X_test, y_test):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')

        train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt', return_attention_mask=True)
        val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt', return_attention_mask=True)
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt', return_attention_mask=True)

        train_dataset = CustomDataset(train_encodings, y_train)
        val_dataset = CustomDataset(val_encodings, y_val)
        test_dataset = CustomDataset(test_encodings, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, device

    def save_checkpoint(self, model, optimizer, epoch, loss, accuracy, path):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, model, optimizer, path, device):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        return model, optimizer, epoch, loss, accuracy

    def train_model(self, model, train_loader, val_loader, optimizer, scheduler, criterion, device, start_epoch, last_loss, last_accuracy):
        n_epochs = self.epochs + start_epoch
        model.to(device)

        # Tambahkan lapisan dropout dan regularisasi L2 pada model
        for name, param in model.named_parameters():
            if 'classifier' not in name:  # Skip classifier layer
                param.requires_grad = True
                if 'bias' in name:
                    modified_name = name.replace('bias', 'weight')
                    if hasattr(model, modified_name):
                        setattr(model, modified_name, nn.Parameter(getattr(model, name).data.clone()))

        # Early stopping
        early_stopping_counter = 0
        best_val_accuracy = last_accuracy
        best_val_loss = last_loss
        best_val_epoch = 0

        for epoch in range(start_epoch + 1, n_epochs + 1):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs}'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device).long()

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()

                optimizer.step()
                scheduler.step()

                clip_grad_norm_(model.parameters(), max_norm=self.gradient_clip_value)

            average_loss = total_loss / len(train_loader)
            self.train_loss_history.append(average_loss)
            print(f'Epoch {epoch}/{n_epochs}, Training Loss: {average_loss}')

            model.eval()
            all_val_predictions = []
            all_val_labels = []
            val_loss = 0

            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc='Evaluating on Validation Data'):
                    val_input_ids = val_batch['input_ids'].to(device)
                    val_attention_mask = val_batch['attention_mask'].to(device)
                    val_labels = val_batch['labels'].to(device).long()

                    val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
                    val_loss += val_outputs.loss.item()  # Accumulate validation loss
                    val_predictions = torch.argmax(val_outputs.logits, dim=1).cpu().numpy()
                    all_val_predictions.extend(val_predictions)
                    all_val_labels.extend(val_labels.cpu().numpy())

            val_accuracy = accuracy_score(all_val_labels, all_val_predictions)
            self.val_accuracy_history.append(val_accuracy)
            val_loss /= len(val_loader)  # Calculate average validation loss
            self.val_loss_history.append(val_loss)
            print(f'Epoch {epoch}/{n_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

            if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and val_loss < best_val_loss):
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                best_val_epoch = epoch
                early_stopping_counter = 0
                self.save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, path=self.model_path+"/checkpoint.pth")
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= self.early_stopping_patience:
                print(f'Early stopping at epoch {epoch}.')
                break ## Mencegah overfitting

        print(f'Akurasi terbaik: {best_val_accuracy} pada epoch ke-{best_val_epoch}')
    
    def plot_learning_curve(self):
        epochs = range(1, len(self.train_loss_history) + 1)

        plt.figure(figsize=(12, 6))

        # Plot loss history
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss_history, label='Training Loss', marker='o')
        plt.plot(epochs, self.val_loss_history, label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy history
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.val_accuracy_history, label='Validation Accuracy', marker='o', color='r')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.image_path, 'learning_curve.png'), dpi=300)
        plt.clf()
        plt.close()

    def evaluate_model(self, model, test_loader, label_encoder, device, optimizer, checkpoint_path):
        model.to(device)
        model.eval()
        all_test_predictions = []
        all_labels = []

        model, _, _, _, _ = self.load_checkpoint(model, optimizer, checkpoint_path, device)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Predicting on Test Data'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                test_predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_test_predictions.extend(test_predictions)
                all_labels.extend(labels.cpu().numpy())

        # Konversi label kembali ke bentuk semula
        decoded_test_predictions = self.decode_labels(all_test_predictions)
        decoded_labels = self.decode_labels(all_labels)

        # Buat confusion matrix
        def plot_matrix():
            cm = confusion_matrix(decoded_labels, decoded_test_predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
            plt.xlabel('Prediksi')
            plt.ylabel('Realitas')
            plt.title('Confusion Matrix')        
            plt.savefig(os.path.join(self.image_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()

        plot_matrix

        # Buat Laporan Klasifikasi
        report = classification_report(decoded_labels, decoded_test_predictions, output_dict=True)
        joblib.dump(report, os.path.join(self.model_path, 'classification_report.joblib'))

    def save_model(self, model, tokenizer, config):
        save_path = self.model_path
        tokenizer.save_pretrained(save_path)
        config.save_pretrained(save_path)
        model.save_pretrained(save_path)

    def training(self):
        self.create_directories()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        config = AutoConfig.from_pretrained(self.pretrained_model)

        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()

        model = BertForSequenceClassification.from_pretrained(self.pretrained_model, num_labels=len(self.label_encoder.classes_))

        train_loader, val_loader, test_loader, device = self.prepare_dataloaders(tokenizer, X_train, y_train, X_val, y_val, X_test, y_test)

        model = model.to(device)

        if self.optimizer_name == 'AdamW':
            optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        elif self.optimizer_name == 'Adam':
            optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        elif self.optimizer_name == 'SparseAdam':
            optimizer = SparseAdam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'Adamax':
            optimizer = Adamax(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Optimizer tidak valid")

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.scheduler_warmup_steps, num_training_steps=len(train_loader)*self.epochs)
        criterion = torch.nn.CrossEntropyLoss()

        checkpoint_path = os.path.join(self.model_path, "checkpoint.pth")

        if os.path.exists(checkpoint_path):
            model, optimizer, start_epoch, loss, accuracy = self.load_checkpoint(model, optimizer, checkpoint_path, device)
            self.load_label_info()
            print(f"Memuat model dari epoch {start_epoch} dengan loss {loss} dan akurasi {accuracy}")
        else:
            start_epoch, accuracy, loss = 0, 0.0, 0.0

        self.train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, start_epoch, loss, accuracy)

        # self.plot_learning_curve()

        self.plot_learning_curve()
        self.evaluate_model(model, test_loader, self.label_encoder, device, optimizer, checkpoint_path)

        self.save_model(model, tokenizer, config)

class IndoBERTClassify:
    def __init__(self, mode, max_length=64):
        self.model_path = os.path.join(model_path, 'indobert', mode)
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.label_encoder = None

    def load_model(self):
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Load label encoder
        label_encoder_path = os.path.join(self.model_path, 'i2w.joblib')
        self.label_encoder = joblib.load(label_encoder_path)

    def predict(self, text_list):
        if not self.model:
            self.load_model()

        # Tokenize text
        inputs = self.tokenizer(text_list, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")

        # Buat prediksi
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)*100

        predictions = torch.argmax(logits, dim=1).item()

        # Decode prediksi menggunakan label_encoder
        decoded_predictions = {
            'label': self.label_encoder['INDEX2LABEL'][predictions],
            'probability': f"{probabilities[0][predictions].item():.2f}%",
            'vector': logits
        }
        return decoded_predictions
    

##################################
##    SUPPORT VECTOR MACHINE    ##
##################################

from app.helpers.preprocessing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import learning_curve, StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import re

from app.model import Negasi

class SVMModel:
    def __init__(self, mode, X_train, X_test, y_train, y_test, n_splits=5, 
                 model_path=model_path, image_path=image_path, svm_kernel='linear', svm_c=1, svm_gamma=1, svm_class_weight=None):
        self.model_path = os.path.join(model_path, 'svm', mode)
        self.image_path = os.path.join(image_path, 'svm', mode)
        self.model_file = os.path.join(self.model_path, 'model.joblib')
        self.params_file = os.path.join(self.model_path, 'params.joblib')
        self.corpus_file = os.path.join(self.model_path, 'corpus.joblib')
        self.vocabs_file = os.path.join(self.model_path, 'vocabs.txt')
        self.vectorizer = TfidfVectorizer()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_splits = n_splits
        self.svm_kernel = svm_kernel
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma
        self.svm_class_weight = svm_class_weight

    def create_directories(self):
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)

    def negation_handling(self, text):
        negasi = [word for word, in Negasi.query.with_entities(Negasi.kata_negasi).all()]
        # Membuat pola regex untuk menemukan kata-kata negasi
        negation_pattern = re.compile(r'\b(?:' + '|'.join(negasi) + r')\b\s+(\w+)')
        # Mengganti kata dengan menambahkan "NOT_"
        result = negation_pattern.sub(lambda match: 'NEGASI_' + match.group(1), text)
        return result

    def save_vocabs(self, feature_names):
        with open(self.vocabs_file, "w") as f:
            for feature in feature_names:
                f.write("%s\n" % feature)

    def training(self):
        self.create_directories()
        combined_model = None

        X_train, y_train = self.X_train, self.y_train

        if os.path.exists(self.corpus_file):
            corpus, labels = joblib.load(self.corpus_file)
            X_train += list(corpus)
            y_train += list(labels)

        joblib.dump((X_train, y_train), self.corpus_file)

        model = make_pipeline(
            TfidfVectorizer(),
            SVC(C=self.svm_c, kernel=self.svm_kernel, gamma=self.svm_gamma, class_weight=self.svm_class_weight)
        )

        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(np.array(X_train).reshape(-1, 1), np.array(y_train))
        X_train = X_train.flatten()

        label_encoder = LabelEncoder()
        y_test_encoded = label_encoder.fit_transform(self.y_test)

        min_class_member = min(np.bincount(y_test_encoded))
        n_splits = min(self.n_splits, min_class_member)

        stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # stratified_shuffle_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        with tqdm(total=n_splits, desc="Training Folds", unit="fold") as pbar:
            for fold, (train_index, test_index) in enumerate(stratified_kfold.split(X_train, y_train)):
                X_train_fold, X_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index]
                y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]
                try:
                    # Lakukan proses negasi
                    X_train_fold = [self.negation_handling(text) for text in X_train_fold]
                    X_test_fold = [self.negation_handling(text) for text in X_test_fold]
                
                    # Melatih model dengan data training
                    model.fit(X_train_fold, y_train_fold)

                    # Menampilkan skor akurasi
                    accuracy = accuracy_score(y_test_fold, model.predict(X_test_fold))
                    print(f"\nFold {fold+1}/{n_splits} ","Akurasi: {:.2f}%".format(accuracy * 100))

                    combined_model = self.update_model(combined_model, model, fold)

                    # Update progress bar
                    pbar.update(1)
                except ValueError as ve:
                    print(f"Error in Fold {fold+1}: {ve}")
                    print(f"Unique values in y_train_fold: {np.unique(y_train_fold)}")

        self.plot_learning_curve(combined_model, self.X_test, self.y_test, n_splits)
        self.evaluate_model(combined_model, self.X_test, self.y_test)

        # Mendapatkan nama fitur (kata-kata) dari TF-IDF Vectorizer
        feature_names = combined_model.steps[0][1].get_feature_names_out()
        self.save_vocabs(feature_names)

        # Menyimpan model gabungan
        joblib.dump(combined_model, self.model_file)

        # Menyimpan model
        joblib.dump(model, self.model_file)

    def update_model(self, combined_model, new_model, fold):
        if combined_model is None:
            return new_model
        else:
            # Menggabungkan model menggunakan rata-rata sederhana
            combined_model.named_steps['svc'].dual_coef_ += new_model.named_steps['svc'].dual_coef_

            print(f"Fold {fold+1} - Model Updated")

            return combined_model

    def evaluate_model(self, model, X, y):
        y_pred = model.predict(X)
        class_labels = model.steps[1][1].classes_

        report = classification_report(y, y_pred, output_dict=True, zero_division=1)
        joblib.dump(report, os.path.join(self.model_path, 'classification_report.joblib'))

        def plot_matrix():
            cm = confusion_matrix(y, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=class_labels, yticklabels=class_labels)
            plt.xlabel("Prediksi")
            plt.ylabel("Realitas")
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(self.image_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()

        plot_matrix()

    def plot_learning_curve(self, model, X, y, n_splits):
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            model, X, y, cv=n_splits, n_jobs=-1,
            train_sizes=np.linspace(.1, 1.0, 5), return_times=True)

        # Hitung nilai rata-rata dan deviasi standar dari skor
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot kurva pembelajaran
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("K-fold Cross-validation Data")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.title("Learning Curve")
        plt.savefig(os.path.join(self.image_path, 'learning_curve.png'), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

class SVMClassify:
    def __init__(self, mode):
        self.model_path = os.path.join(model_path, 'svm', mode)
        self.model_file = os.path.join(self.model_path, 'model.joblib')
        self.svm_model = SVMModel(mode, None, None, None, None)

    def predict(self, text):
        model = joblib.load(self.model_file)
        text = preprocessing(text)
        text = self.svm_model.negation_handling(text)
        vectorizer = model.steps[0][1]
        vectorized_text = vectorizer.transform([text])
        non_zero_indices = vectorized_text.nonzero()[1]
        feature_names = vectorizer.get_feature_names_out()

        vectors = []
        for index in non_zero_indices:
            feature = feature_names[index]
            weight = vectorized_text[0, index]
            vector = f"{feature}: {weight:.4f}"
            vectors.append(vector)

        prediction = model.predict([text])

        result = {
            'label': prediction[0],
            'vector': vectors
        }

        return result