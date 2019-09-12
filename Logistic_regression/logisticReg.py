# -*- coding: utf-8 -*-

import xml.etree.cElementTree as ET
import pandas as pd


utter_id_lt = []
utter_lng_lt = []

train_utter_id = []
train_utter_lng = []
with open("AnnotationTraining.xml",encoding="utf8") as fp:
    data = fp.read().replace("&","|")
    Annot = ET.fromstring(data)
    ##getting utterance tag
    for utterance in Annot.findall('utterance'):
        utter_id= utterance.attrib['id'].strip()
        utter_lng = utterance.text.strip()
        utter_id_lt.append(utter_id)
        utter_lng_lt.append(utter_lng)
with open("InputTraining.xml",encoding="utf8") as fp:
    data = fp.read().replace("&","|")
    inputTrain = ET.fromstring(data)
    ##getting utterance tag
    for utterance in inputTrain.findall('utterance'):
        utter_id= utterance.attrib['id'].strip()
        utter_lng = utterance.text.strip()
        train_utter_id.append(utter_id)
        train_utter_lng.append(utter_lng)
names = ["an_id", "an_txt", "tr_id","tr_txt"]
final_df = pd.DataFrame(columns=names)
final_df["an_id"] = utter_id_lt
final_df["an_txt"] = utter_lng_lt
final_df["tr_id"] = train_utter_id
final_df["tr_txt"] = train_utter_lng

final_df["an_word_cnt"] = final_df["an_txt"].apply(lambda x : len(x.split(" ")))
final_df["tr_word_cnt"] = final_df["tr_txt"].apply(lambda x :len(list(filter(None,x.split(" ")))))  
##checking the miss match mapping 
check_cnt = final_df[final_df["an_word_cnt"] != final_df["tr_word_cnt"]]


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(final_df["tr_txt"], final_df["an_txt"])




        

        
        
    
