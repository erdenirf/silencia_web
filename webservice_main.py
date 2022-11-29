import gensim
import logging
import re
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from erdeni_nlp import DataCleaning, Preprocessing, Tokenization, Lemmatization
from zipfile import ZipFile
import pymorphy2
import json
import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips
from annotated_text import annotated_text
import wget
import os

# Загрузить словарь
with open('vocabulary.json', "r", encoding='utf-8') as f:
    vocabulary = json.loads(f.read())
with open('vocabulary_not_in_model.json', "r", encoding='utf-8') as f:
    not_in_model = json.loads(f.read())
with open('stopwords.json', "r", encoding='utf-8') as f:
    stopwords_drops = json.loads(f.read())
the_keep_stopwords = list(map(lambda x: x.split("_")[0], stopwords_drops.keys()))
    
# Инстанс лемматизатора
morph = pymorphy2.MorphAnalyzer()

# Инстанс модели word2vec
model_url = 'http://vectors.nlpl.eu/repository/20/220.zip'
model_file = model_url.split('/')[-1]
folder_name = model_file.split('.')[0]

file_exists = os.path.exists(model_file)
if not file_exists:
    m = wget.download(model_url)

folder_exists = os.path.exists(folder_name)
if not folder_exists:
    zf = ZipFile(model_file, 'r')
    zf.extractall(folder_name)
    zf.close()
model = gensim.models.KeyedVectors.load_word2vec_format("220/model.bin", binary=True)


# Webservice BEGIN

class Translater:

    def __init__(self, morph=morph, the_keep_stopwords=the_keep_stopwords):
        self.morph = morph
        self.the_keep_stopwords = the_keep_stopwords

    def new_calculation(self, input_text):
        input_filtered = []
        output_lemma = []
        output_similarity = []
        annotated_args = []

        IS_STOPWORD = "stopword"
        IS_FOUND = "found"

        for word in Tokenization(Preprocessing(DataCleaning(input_text))):
            code, lemma = Lemmatization(self.morph, word, keep_pos=True, convert_upos=True, 
                                        drop_stopword=True, keep_stopwords=self.the_keep_stopwords)
            if not code:
                continue
            
            stopword_filename = stopwords_drops.get(lemma)
            if stopword_filename:
                input_filtered.append(word)
                output_lemma.append(lemma)
                output_similarity.append(IS_STOPWORD)
                annotated_args.append(" " + word + " ")
                continue
            
            similarity_max = 0
            similarity_word = None
            for iterated_word in vocabulary.keys():
                if lemma == iterated_word:
                    #filename = vocabulary.get(lemma)
                    input_filtered.append(word)
                    output_lemma.append(lemma)
                    output_similarity.append(IS_FOUND)
                    annotated_args.append(" " + word + " ")
                    similarity_word = None
                    break
                    
                if iterated_word in not_in_model:
                    continue

                if lemma in model:
                    percent = model.similarity(lemma, iterated_word)
                    if percent > similarity_max:
                        similarity_max = percent
                        similarity_word = iterated_word
            
            if similarity_word:
                #filename = vocabulary.get(similarity_word)
                input_filtered.append(word)
                output_lemma.append(similarity_word)
                output_similarity.append(similarity_max)
                annotated_args.append((word, similarity_word + " " + str(similarity_max)))

        st.session_state.text_output = " ".join(list(map(lambda x: x.split("_")[0], output_lemma)))
        st.session_state.annotations = annotated_args
        return None

    def get_output(self):
        if "text_output" in st.session_state:
            return st.session_state.text_output
        return None

    def get_annotations(self):
        if "annotations" in st.session_state:
            return st.session_state.annotations
        return None

translater = Translater()


HEIGHT_TEXTAREA = 200
INPUT_LABEL = 'Входной текст на русском языке:'
OUTPUT_LABEL = "Перевод на жестовый словарь:"

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
</style>""", unsafe_allow_html=True)

text_input = st.text_area(INPUT_LABEL, "", height=HEIGHT_TEXTAREA)

text_area = st.empty()

text = text_area.text_area(OUTPUT_LABEL, "", height=HEIGHT_TEXTAREA)

annotations = st.empty()


button_translate = st.button('Перевести')
if button_translate:
    
    translater.new_calculation(text_input)
    output_text = translater.get_output()
    if output_text:
        text = text_area.text_area(OUTPUT_LABEL, output_text, height=HEIGHT_TEXTAREA)
        st.write("\n")
    annots = translater.get_annotations()
    if annots:
        annotations = annotated_text(annots)
        st.write("\n")
else:
    st.write('Нажмите перевести')


button_video = st.button('Запустить видео')
if button_video:

    output_text = translater.get_output()
    if output_text:
        text = text_area.text_area(OUTPUT_LABEL, output_text, height=HEIGHT_TEXTAREA)
        st.write("\n")
    annots = translater.get_annotations()
    if annots:
        annotations = annotated_text(annots)
        st.write("\n")

    if text:

        #clips = [VideoFileClip(c) for c in ['адрес.mp4']]
        #final_clip = concatenate_videoclips(clips)
        #final_clip.write_videofile("final.mp4")
        video_file3 = open('адрес.mp4', 'rb')
        video_bytes3 = video_file3.read()
        st.video(video_bytes3, format="video/mp4")
    else:
        st.write('Для запуска видео, сначала нажмите кнопку "Перевести"')