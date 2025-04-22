# https://docs.python.org/3/howto/unicode.html
# commands to setup this code
#python3 -m pip install tensorflow
#pip install nltk
#pip install bs4
#pip install matplotlib
#pip install langdetect
#python3 -m pip install python-geoip-python3
#python3 -m pip install python-geoip-geolite2

import tensorflow as tf
import pandas as pd
import numpy as np
import nltk
# nltk.download('omw-1.4')
import re
from bs4 import BeautifulSoup
import os
import quopri
import email
from email.header import decode_header, make_header
import base64
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras import models, layers, preprocessing as kprocessing, Sequential
import glob
import gc
import random
import threading
import queue
import boto3
import csv
from langdetect import detect
import geoip2
import geoip2.database
import json


EMAIL_PATH = 'raw_emails/'
EMAIL_PATH_EXTRACTED = 'emails_with_subject_body/'

#create folder if not already
if not os.path.exists(EMAIL_PATH_EXTRACTED):
    os.makedirs(EMAIL_PATH_EXTRACTED)

# clean up the output folder
files = glob.glob(EMAIL_PATH_EXTRACTED+'/*')
for f in files:
    os.remove(f)

class PreProcessor:

    def __init__(self):
        nltk.download("wordnet", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        self.lst_stopwords = nltk.corpus.stopwords.words("english")
        self.lem = nltk.stem.wordnet.WordNetLemmatizer()
#         self.max_sequence_length = 300

#         with open(os.getenv("TOKENIZER_PICKLE_PATH"), "rb") as tokenizer_file:
#             self.tokenizer = pickle.load(tokenizer_file)

    # fully preprocesses a given raw mime body text to tokenized sequence
    def process(self, text):
        subject_body = self._pull_subject_body(text)
        stripped_html = self._strip_html(subject_body)
        cleaned = self._clean(stripped_html)
        no_stopwords = self._remove_stop_words(cleaned)
        lemmatized = self._lemmatize(no_stopwords)
#         tokenized = self.tokenizer.texts_to_sequences([lemmatized])
#         return kprocessing.sequence.pad_sequences(
#             tokenized,
#             maxlen=self.max_sequence_length,
#             padding="post",
#             truncating="post")
        return lemmatized


    def _pull_subject_body(self, text):
        a = email.message_from_string(text)
        body = None
        
        if a.is_multipart():
            for content in a.walk():
                if not content.is_multipart() and (content.get_content_type() == "text/html"):
                    try:
                        soup = BeautifulSoup(content.get_payload(), 'html.parser')
                    except:
                        soup = BeautifulSoup(content.get_payload(), 'lxml')
                    cs = content.get_content_charset() if content.get_content_charset() else "utf-8"
                    htmlCS = soup.meta.get("charset").strip("3D").strip("=").strip("//").strip("\"").strip(";") if soup.meta and soup.meta.get("charset") else cs
                    
                    payload = content.get_payload()
                    if content['Content-Transfer-Encoding'] is not None and content['Content-Transfer-Encoding'].strip() == "base64":
                        if body == None or body == '':
                            body = self._try_html_vs_root_cs("_decode_base64", payload, cs, htmlCS) #if body is None or body is '' else body
                    elif content['Content-Transfer-Encoding'] is not None and content['Content-Transfer-Encoding'].strip() == "quoted-printable":
                        if body == None or body == '':
                            body = self._try_html_vs_root_cs("_decode_quoted_printable", payload, cs, htmlCS) #if body is None or body is '' else body
                    else:
                        if body == None or body == '':
                            body = payload # if body is None else body #.decode('UTF-8')


        else:
            try:
                soup = BeautifulSoup(a.get_payload(), 'html.parser')
            except:
                soup = BeautifulSoup(a.get_payload(), 'lxml')
            cs = a.get_content_charset() if a.get_content_charset() else "utf-8"
            htmlCS = soup.meta.get("charset").strip("3D").strip("=").strip("//").strip("\"").strip(";") if soup.meta and soup.meta.get("charset") else cs

            payload = a.get_payload()
            if a['Content-Transfer-Encoding'] is not None and a['Content-Transfer-Encoding'].strip() == "base64":
                if body == None or body == '':
                    body = self._try_html_vs_root_cs("_decode_base64", payload, cs, htmlCS)# if body is None or body is '' else body
                    
            elif a['Content-Transfer-Encoding'] is not None and a['Content-Transfer-Encoding'].strip() == "quoted-printable":
                if body == None or body == '':
                    body = self._try_html_vs_root_cs("_decode_quoted_printable", payload, cs, htmlCS)# if body is None or body is '' else body
            else:
                if body == None or body == '':
                    body = payload # if body is None else body #.decode('UTF-8')

        subject = a['Subject']
        if body == None:
            body = ''
        if subject:
            try:
                subject = str(make_header(decode_header(subject)))
            except LookupError:
                subject = subject
            return (subject + " " + body, body)

        return (body, body)

    def _decode_base64(self, payload, cs):
        return base64.b64decode(bytes(payload, encoding=cs, errors="backslashreplace")).decode(cs, errors='backslashreplace') 

    def _decode_quoted_printable(self, payload, cs):
        return quopri.decodestring(bytes(payload, encoding=cs, errors="backslashreplace"),header = False).decode(cs, errors='backslashreplace') #backslashreplace
    
    def _decode_8bit(self, payload, cs):
        return payload

    def _try_html_vs_root_cs(self, type, payload, htmlcs, cs):
        try:
            body = getattr(self, type)(payload, htmlcs)
        except LookupError:
            body = getattr(self, type)(payload, cs)
        return body

    # removes all html tags
    def _strip_html(self, text):
        # create queue for html strip proc
        q = queue.Queue()

        # strip html in thread to handle memory leak in lxml - mem is freed when thread is joined
        stripper = stripHtml(text, q)
        stripper.start()
        stripper.join(2)
        stripped = q.get()

        # return stripped text
        return stripped

    # removes all newlines, punctuations, and characters and converts to lowercase
    def _clean(self, text):
        lst = text.splitlines()
        new_list = []
        for s in lst:
            if s == '':
                continue
            new_list.append(s)
        no_newlines = " ".join(new_list)
        return re.sub(r'[^\w\s]', ' ', str(no_newlines).lower().strip())

    def cleaned_html(self, text):
        stripped = self._strip_html(text)
        cleaned = self._clean(stripped)
        return cleaned

    # removes all stop words defined by the nltk package
    def _remove_stop_words(self, text):
        lst_text = text.split()
        lst_no_stopwords = [
            word for word in lst_text if word not in self.lst_stopwords
        ]
        return " ".join(lst_no_stopwords)

    # completes lemmatization using nltk package
    def _lemmatize(self, text):
        lst_text = text.split()
        lst_lemmatized = [self.lem.lemmatize(word) for word in lst_text]
        return " ".join(lst_lemmatized)


# thread class implementation for stripHtml
class stripHtml(threading.Thread):

    def __init__(self, text, q):
        threading.Thread.__init__(self)
        self.text = text
        self.q = q

    def run(self):
        soup = BeautifulSoup(self.text, 'lxml')
        stripped = soup.get_text()
        self.q.put(stripped)

        # explicitly destroy the object to handle known memory leak in underlying lxml parser, mem freed when thread is terminated
        soup.decompose()
        del soup

# p = PreProcessor()


# user_id_list = []
# msg_id_list = []
# sg_event_id_list = []
# subject_list = []
# from_email_list = []
# to_email_list = []
# email_date_list = []
# orig_ip_list = []
# orig_ip_country_list = [] #using Maxmind
# lang_list = [] #using langdetect

# for filename in os.listdir(EMAIL_PATH):
#     path = os.path.join(EMAIL_PATH, filename)
#     if os.path.isfile(path):
#         f = open(path, "r")
#         j = json.loads(f.read())
#         raw_mime = j["raw_mime"]
#         subject_body = p._pull_subject_body(raw_mime)
#         f.close()
        
#         f = open(EMAIL_PATH_EXTRACTED + filename + ".html", "w")
#         f.write(subject_body)
#         f.close()

#         try:
#             user_id = j["event"]["payload"]["userid"]

#             msg_id = j["event"]["payload"]["msgid"]
#             sg_event_id = j["event"]["payload"]["sg_event_id"]
#             subject = j["event"]["payload"]["subject"]
#             if subject:
#                 subject = str(make_header(decode_header(subject)))

#             email_from = j["event"]["payload"]["email_from"]
#             email_to = j["event"]["payload"]["email"]
#             email_date = j["event"]["payload"]["date"]
#             originating_ip = j["event"]["payload"]["originating_ip"]
#             originating_ip_country = 'N/A'
#             try:
#                 with open_database('./GeoIP2-Country.mmdb') as db:
#                     match = db.lookup(originating_ip)
#                     originating_ip_country = match.country
#             except Exception as e: 
#                 print('[ERROR]', e)

#             lang = 'N/A'
#             try:
#                 lang = detect (subject)
#             except:
#                 None

#             user_id_list.append(user_id)
#             msg_id_list.append(msg_id)
#             sg_event_id_list.append(sg_event_id)
#             subject_list.append(subject)
#             from_email_list.append(email_from)
#             to_email_list.append(email_to)
#             email_date_list.append(email_date)
#             orig_ip_list.append(originating_ip)
#             orig_ip_country_list.append(originating_ip_country)
#             lang_list.append(lang)
#         except:
#             print ( "[ERROR] Failed to process! ", sg_event_id)
#             error_count += 1

# df = pd.DataFrame()            
# df["user_id"] = user_id_list
# df["msg_id"] = msg_id_list
# df["sg_event_id"] = sg_event_id_list
# df["subject"] = subject_list
# df["email_from"] = from_email_list
# df["email_to"] = to_email_list
# df["email_date"] = email_date_list
# df["originating_ip"] = orig_ip_list
# df["originating_ip_country"] = orig_ip_country_list
# df["lang"] = lang_list

# df = df.sort_values(['user_id', 'sg_event_id'])

# df.to_csv('email_tagging_judgement_sheet.csv',index=False)