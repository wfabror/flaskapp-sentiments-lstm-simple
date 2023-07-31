from keras import backend as K
from datetime import datetime
import sys
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
import tensorflow as tf
import operator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import time
import re
import io
from flask import *
import os

app = Flask(__name__)

nltk.download('stopwords')
pd.set_option("display.max_columns", None)


SAVED_FOLDER = './saved-files'
app.config['SAVED_FOLDER'] = SAVED_FOLDER
# start here
df_train = pd.read_csv(os.path.join(
    app.config['SAVED_FOLDER'], 'translated_data_latih_new.csv'))
df_uji_900 = pd.read_csv(os.path.join(
    app.config['SAVED_FOLDER'], 'data_uji.csv'))
df_uji_1400 = pd.read_csv(os.path.join(
    app.config['SAVED_FOLDER'], 'datauji_vaksin_1400.csv'))
df_uji_2100 = pd.read_csv(os.path.join(
    app.config['SAVED_FOLDER'], 'datauji_vaksin_2100.csv'))

df_train_head = df_train.head()
df_uji_900_head = df_uji_900.head()
df_uji_1400_head = df_uji_1400.head()
df_uji_2100_head = df_uji_2100.head()

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H")


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" %
                   (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def data_cleaning(input_):
    sentiment = input_[['Text']]
    df = sentiment.copy()
    # case-folding
    print('Process 1/6 - CaseFolding')
    for i in progressbar(range(15), "Computing: ", 40):
        case_fold = df['Text'].str.casefold()

    # cleaning unnecessary elements
    print('Process 2/6 - DataCleaning')
    for i in progressbar(range(15), "Computing: ", 40):
        data_clean = case_fold.apply(lambda x: re.sub(
            r"(?:\@|http?\://|https?\://|www)\S+", '', str(x)))
        data_clean = data_clean.str.replace('\n', ' ')
        data_clean = data_clean.str.replace(r'[^\w\s]+', ' ', regex=True)
        data_clean = data_clean.str.replace(r"\d+", '', regex=True)
        data_clean = data_clean.apply(lambda x: x.strip())
    return data_clean


def text_processing(input_):
    # Text Tokenization
    print('Process 3/6 - Tokenization')
    for i in progressbar(range(15), "Computing: ", 40):
        word_tokens = []
        for i in range(len(input_)):
            tokenizer = word_tokenize(input_[i])
            word_tokens.append(tokenizer)

    # Abbreviation Vocab
    abbv = ["tdk", "gak", "ngga", "ga", "yg", "emng", "mmng", "knp", "stlh", "gara",
            "krn", "hrs", "msh", "bkn", "yaa", "trs", "sdh", "untk", "dgn", "mksd",
            "gk", "y", "thn", "jd", "skrg", "sampe", "bapakk", 'dlm', 'cuuuy', "yg ",
            'tak', 'kalo', 'sekrng', 'kek', 'gue', 'sya', "kpd", 'alia', 'ama', 'jg',
            'kmrn', 'dapet', 'bgt', 'org', 'emang', 'tapi', 'rs', 'mikir', 'case', 'klo',
            'mash', 'udah', 'lg', 'cewe', 'biar', 'pukul', 'nda', 'bs', 'enggak', 'aja',
            'gitu', 'cuma', 'malem', 'gmn', 'ahir', 'mbantu', 'online', 'bener', 'tetep',
            'ngeyel', 'g', 'lgi', 'banget', 'tak', 'udh', 'bar', 'kabeh', 'opo', 'nek',
            'deket', 'taruk', 'sini', 'jgn', 'duit', 'kelar', 'ttep', 'kosan', 'tp', 'joko',
            'widodo', 'mesjid', 'kaulah', 'ki', 'ilang', 'lakok', 'belehe', 'do', 'viruse',
            'gw', 'kec', 'kel']
    conv = ["tidak", "tidak", "tidak", "tidak", "yang", "memang", "memang", "kenapa",
            "setelah", "karena", "karena", "harus", "masih", "bukan", "ya", "terus",
            "sudah", "untuk", "dengan", "maksud", "tidak", "ya", "tahun", "jadi",
            "sekarang", "sampai", "bapak", 'dalam', '', "yang ", 'tidak', 'kalau',
            'sekarang', 'seperti', 'kamu', 'saya', 'kepada', "alias", 'sama', 'juga',
            'kemarin', 'dapat', 'sangat', 'orang', 'memang', 'tetapi', 'rumah sakit',
            'berpikir', 'kasus', 'kalau', 'masih', 'sudah', 'lagi', 'perempuan', 'supaya',
            'jam', 'tidak', 'bisa', 'tidak', 'saja', 'begitu', 'hanya', 'malam', 'bagaimana',
            'akhir', 'membantu', 'daring', 'benar', 'tetap', 'bandel', 'tidak', 'lagi',
            'sangat', 'tidak', 'sudah', 'setelah', 'semua', 'apa', 'kalau', 'dekat', 'taruh',
            'disini', 'jangan', 'uang', 'selesai', 'tetap', 'kos', 'tapi', 'jokowi', 'jokowi',
            'masjid', 'kau', 'itu', 'hilang', 'kok', 'qurban', 'pada', 'virusnya', 'saya',
            'kecamatan', 'kelurahan']

    # Text Lemmatization
    print('Process 4/6 - Lemmatization')
    for i in progressbar(range(15), "Computing: ", 40):
        lemmatized_data = []
        full_sentence = ''

        for i in range(len(word_tokens)):
            result = []
            for word in word_tokens[i]:
                if word in abbv:
                    indexs = abbv.index(word)
                    converted = conv[indexs]
                    result.append(converted)
                else:
                    result.append(word)

            full_sentence = ' '.join(result)
            lemmatized_data.append(full_sentence)

    # Removing Stopwords
    print('Process 5/6 - Removing Stopwords')
    for i in progressbar(range(15), "Computing: ", 40):
        stop_factory = StopWordRemoverFactory().get_stop_words()
        more_stopword = ['dg', 'amp', 'urg', 'wkw', 'yg', 'sia', 'xx',
                         'n', 'kan', 'diy', 'sy', 'tsb', 'lalu', 'bang',
                         'pak', 'kak', 'rangkasbitung', 'mak', 'its',
                         'allahu', 'm', 'kenza', 'won', 'sh', 'oma', 'di',
                         'gin', 'winning', 'pas', 'door', 'ku', 'aku',
                         'kamu', 'biar', 'yah', 'eh', 'ra', 'akan', 'fix',
                         'rt', 'pp', 'h', 'kita', 'nang', 'guys', 'nder',
                         'iya', 'we', 'to', 'an', 'bong', 'herlin',
                         'nih', 'kan', 'anjing', 'njing', 'bu', 'bat',
                         'gue', 'in', 'artisesungguhnya', 'keadaansesungguhnyanegaraini',
                         'j', 'int', 'gada', 'kek', 'dong', 'mas', 'so', 'on', 'si', 'gw',
                         'too', 'and', 'lah', 'well', 'fren', 'sih', 'sksk', 'anjay', 'dm',
                         'mah', 'mbok', 'rw']

        # Merge stopword
        data = stop_factory + more_stopword

        dictionary = ArrayDictionary(data)
        str = StopWordRemover(dictionary)

        no_stopwords = []
        for i in lemmatized_data:
            stop = str.remove(i)
            no_stopwords.append(stop)

        df_final = pd.DataFrame(no_stopwords, columns=['Tweets'])
        df_final = df_final.drop_duplicates().reset_index(drop=True)
        df_final = df_final.apply(lambda x: x.str.strip()).replace(
            '', np.nan).dropna().reset_index(drop=True)

    return df_final


def predict_label(input_, analysis, unseen_data):
    print('Process 7/7 - Predicting Label')
    # Encoding the target column
    lb = LabelEncoder()
    input_['analysis_encoded'] = lb.fit_transform(input_[analysis])

    # Tokenizing the parameter
    concat = pd.concat([input_['Tweets'], unseen_data['Tweets']])
    tokenizer = Tokenizer(num_words=500, split=' ')
    tokenizer.fit_on_texts(concat.values)
    X = tokenizer.texts_to_sequences(concat.values)
    X = pad_sequences(X)

    # Modelling LSTM
    model = Sequential()
    model.add(Embedding(500, 120, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(176, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(len(input_[analysis].unique()), activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Splitting the data into training and testing
    y = pd.get_dummies(input_[analysis])
    X_train = X[0:1847, :]
    y_train = y.iloc[:1847, :]
    # Starting to train the model
    batch_size = 32
    history = model.fit(X_train, y_train, epochs=10,
                        batch_size=batch_size, verbose='auto')

    predict_x = model.predict(X[1847:len(concat), :])
    label_predict = np.argmax(predict_x, axis=1)
    unseen_data['Predicted'] = label_predict
    unseen_data['Predicted'].replace(
        {0: "Negative", 1: "Positive"}, inplace=True)

    # access validation accuracy for each epoch
    acc = history.history['accuracy']
    acc_df = pd.DataFrame(acc)
    acc_plot = acc_df.plot.line()
    acc_plot.legend(["Train Accuracy per Epoch"])
    acc_plot.figure.savefig('output/train_ltsm.png')

    acc_max = str(round(np.max(history.history['accuracy'])*100, 2))

    return unseen_data, acc, acc_max


def read_acc(input_):
    acc_iter = []
    for item in input_:
        i = round((item*100), 2)
        acc_iter.append(i)
    return acc_iter


def sentiment_analysis(input_):
    def polarity(data):
        blob = TextBlob(data)
        return blob.polarity

    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        else:
            return 'Positive'

    input_['polarity_score'] = input_['Translated'].apply(polarity)
    input_['analysis_result'] = input_['polarity_score'].apply(getAnalysis)
    return input_


def summary(df_ori, df_cleaning, df_textpreprocessing):
    sentiment = df_ori[['Text']]
    df_resume = sentiment.copy()
    df_resume['Cleansing'] = df_cleaning
    df_resume['Text Processing'] = df_textpreprocessing['Tweets']
    df_resume_head = df_resume.head()
    return df_resume_head


# df_clean_train = data_cleaning(df_train)
df_clean_uji900 = data_cleaning(df_uji_900)
df_clean_uji1400 = data_cleaning(df_uji_1400)
df_clean_uji2100 = data_cleaning(df_uji_2100)

# df_tp_train = text_processing(df_clean_train)
df_tp_uji900 = text_processing(df_clean_uji900)
df_tp_uji1400 = text_processing(df_clean_uji1400)
df_tp_uji2100 = text_processing(df_clean_uji2100)


analyze = 'analysis_result'

df_predict_uji900, acc_uji900, acc_max_uji900 = predict_label(df_train, analyze, df_tp_uji900)
df_predict_uji1400, acc_uji1400, acc_max_uji1400 = predict_label(df_train, analyze, df_tp_uji1400)
df_predict_uji2100, acc_uji2100, acc_max_uji2100 = predict_label(df_train, analyze, df_tp_uji2100)

#

@app.route('/')
def abstrak():
    return render_template("abstrak.html")


@app.route('/datasetOri')
def datasetOri():
    data_train_html = df_train_head.to_html(
        classes='table table-hover', justify='justify', index=False)
    data_uji900_html = df_uji_900_head.to_html(
        classes='table table-hover', justify='justify', index=False)
    data_uji1400_html = df_uji_1400_head.to_html(
        classes='table table-hover', justify='justify', index=False)
    data_uji2100_html = df_uji_2100_head.to_html(
        classes='table table-hover', justify='justify', index=False)

    return render_template("datasetOri.html",
                           data_train=data_train_html,
                           data_uji900=data_uji900_html,
                           data_uji1400=data_uji1400_html,
                           data_uji2100=data_uji2100_html
                           )


@app.route('/preprocessing')
def preprocessingResult():
    df_summary_uji900 = summary(df_uji_900, df_clean_uji900, df_tp_uji900).to_html(
        classes='table table-hover', justify='justify', index=False)
    df_summary_uji1400 = summary(df_uji_1400, df_clean_uji1400, df_tp_uji1400).to_html(
        classes='table table-hover', justify='justify', index=False)
    df_summary_uji2100 = summary(df_uji_2100, df_clean_uji2100, df_tp_uji2100).to_html(
        classes='table table-hover', justify='justify', index=False)

    return render_template('preprocessing.html',
                           summary_uji900=df_summary_uji900,
                           summary_uji1400=df_summary_uji1400,
                           summary_uji2100=df_summary_uji2100,
                           )


@app.route('/result')
def modellingResult():
    
    result_uji900 = read_acc(acc_uji900)
    df_result_uji900 = df_predict_uji900.head().to_html(classes='table table-hover', justify='justify', index=False)
    len_acc_uji900 = len(result_uji900)
    
    result_uji1400 = read_acc(acc_uji1400)
    df_result_uji1400 = df_predict_uji1400.head().to_html(classes='table table-hover', justify='justify', index=False)
    len_acc_uji1400 = len(result_uji1400)

    result_uji2100 = read_acc(acc_uji2100)
    df_result_uji2100 = df_predict_uji2100.head().to_html(classes='table table-hover', justify='justify', index=False)
    len_acc_uji2100 = len(result_uji2100)
    
    return render_template('result.html',
                           len_uji900 = len_acc_uji900,
                           res_uji900 = result_uji900,
                           df_res_uji900 = df_result_uji900,
                           top_acc_uji900 = acc_max_uji900,
                           len_uji1400 = len_acc_uji1400,
                           res_uji1400 = result_uji1400,
                           df_res_uji1400 = df_result_uji1400,
                           top_acc_uji1400 = acc_max_uji1400,
                           len_uji2100 = len_acc_uji2100,
                           res_uji2100 = result_uji2100,
                           df_res_uji2100 = df_result_uji2100,
                           top_acc_uji2100 = acc_max_uji2100, 
                           )


#
if __name__ == "__main__":
    app.run(debug=True)
