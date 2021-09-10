import tqdm
import en_core_web_sm
import pandas as pd
import numpy as np
from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


class preprocessor:
    tag_preprocessor = None
    paragraphs = []
    questions = []
    untok_paragraphs = []
    ids = []
    processed_paragraphs = []
    processed_questions = []
    word_tokenizer = None
    paragraph_maxlen = None
    question_maxlen = None
    input_filepath = None
    
    def __init__(self,input_fp,word_tokenizer_fp,ner_tokenizer_fp,pos_tokenizer_fp,paragraph_maxlen,question_maxlen):
        self.input_filepath = input_fp
        self.word_tokenizer = load(open(word_tokenizer_fp,'rb'))
        ner_tokenizer = load(open(ner_tokenizer_fp, 'rb'))
        pos_tokenizer = load(open(pos_tokenizer_fp, 'rb'))
        self.tag_preprocessor = en_core_web_sm.load()
        self.paragraph_maxlen = paragraph_maxlen
        self.question_maxlen = question_maxlen
        self.fetch_input(input_fp)
        self.processed_paragraphs = self.make_padding(self.tokenize(self.word_tokenizer,pos_tokenizer,ner_tokenizer,self.preprocess(self.paragraphs)),self.paragraph_maxlen)
        self.processed_questions = self.make_padding(self.tokenize(self.word_tokenizer,pos_tokenizer,ner_tokenizer,self.preprocess(self.questions)),self.question_maxlen)
        
        
    def fetch_input(self,input_fp):
        with open(input_fp) as dataset_file:
            dataset_json = json.load(dataset_file)
        for data in dataset_json["data"]:
          for paragraph in data["paragraphs"]:
            for qa in paragraph["qas"]:
              question = qa["question"]
              for answer in qa["answers"]:
                self.paragraphs.append(paragraph["context"])
                self.untok_paragraphs.append(paragraph["context"].split(" "))
                self.questions.append(question)
                self.ids.append(qa["id"])

    def preprocess_tags(self,text):
      doc = self.tag_preprocessor(text)
      return [ (t.text, t.tag_, t.ent_type_, t.like_num, t.is_stop) for t in doc ]
    
    def normalize(self,token):
      text, pos, entity, is_num, is_stop = token
      text = text.lower()
      return (text, pos, entity, is_num, is_stop)
    
    def preprocess(self,dataset):
      return [[self.normalize(token) for token in self.preprocess_tags(text)] for text in dataset]
  
    def make_padding(self,dataset, max_len=None):
        return pad_sequences(dataset, padding='post', value=0.0, maxlen=max_len)

    def tokenize(self,tokenizer, pos_tag_to_idx, ner_to_idx, dataset):
      result = []
      for paragraph in dataset:
        result_paragraph = []
        for text, pos, ent, is_num, is_stop in paragraph:
          token = tokenizer.texts_to_sequences([text])[0]
          if text != ' ' and len(token):
            result_paragraph.append((token[0], pos_tag_to_idx[pos], ner_to_idx[ent], is_num, is_stop))
        result.append(result_paragraph)
      return result
  
    def classify(self,dataset,tokenizer):
        classifier = []
        question_classifiers = {"what":1,"how":2,"why":3,"where":4,"when":5,"which":6,"who":7,"whose":8,"whom":9}
        for question in dataset:
          question = tokenizer.sequences_to_texts(question)
          keywords = set(question_classifiers.keys()).intersection(set(question))
          if len(keywords) == 1:
            classifier.append(question_classifiers[keywords.pop()])
          else:
            classifier.append(0)
        return np.array(classifier)
        
    def get_model_input(self):
        X_question_text = np.expand_dims(self.processed_questions[:,:,0], axis=-1)
        X_question_pos = np.expand_dims(self.processed_questions[:,:,1], axis=-1)
        X_question_ner = np.expand_dims(self.processed_questions[:,:,2], axis=-1)
        X_question_is_num = np.expand_dims(self.processed_questions[:,:,3], axis=-1)
        X_question_is_stop = np.expand_dims(self.processed_questions[:,:,4], axis=-1)
        X_paragraph_text = np.expand_dims(self.processed_paragraphs[:,:,0], axis=-1)
        X_paragraph_pos = np.expand_dims(self.processed_paragraphs[:,:,1], axis=-1)
        X_paragraph_ner = np.expand_dims(self.processed_paragraphs[:,:,2], axis=-1)
        X_paragraph_is_stop = np.expand_dims(self.processed_paragraphs[:,:,3], axis=-1)
        X_paragraph_is_num = np.expand_dims(self.processed_paragraphs[:,:,4], axis=-1)
        X_flag_present = np.array([[bool(token and token in question) for token in paragraph] for paragraph,question in zip(X_paragraph_text,X_question_text)])
        X_classifier = self.classify(X_question_text,self.word_tokenizer)
        return  [X_question_text, X_question_pos, X_question_ner, X_question_is_num, X_question_is_stop,X_classifier,
         X_paragraph_text, X_paragraph_pos, X_paragraph_ner, X_paragraph_is_stop, X_paragraph_is_num, X_flag_present]
    

    def map_predictions(self,start_pred,end_pred):
        answers = {}
        paragraph_text = [self.word_tokenizer.sequences_to_texts([[word[0] for word in paragraph]])[0].split() for paragraph in self.processed_paragraphs]
        for i in range(start_pred.shape[0]):
           # offset = int(self.paragraph_maxlen - len(paragraph_text[i]))
            answers[self.ids[i]] = " ".join(paragraph_text[i][start_pred[i]:end_pred[i]+1])
        return answers
    
    
        
   