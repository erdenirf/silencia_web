import gensim
import logging
import re
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from erdeni_nlp import DataCleaning, Preprocessing, Tokenization, Lemmatization
from zipfile import ZipFile
import pymorphy2
import json
import streamlit as st
#from moviepy.editor import VideoFileClip, concatenate_videoclips
from annotated_text import annotated_text
import wget
import os
import time
import tempfile
import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

# mediapipe

class cv2_VideoCapture_from_list:
    
    def __init__(self, filenames: list):
        
        self.array = filenames.copy()
        self.current_index = 0
        self.capture = cv2.VideoCapture(self.array[self.current_index])
        self.names = list(map(lambda x: x.split(".")[0], filenames.copy()))
                
    def isOpened(self):
        if self.current_index >= len(self.array):
            return False
        return self.capture.isOpened()
    
    def get(self, constant):
        if self.current_index >= len(self.array):
            return False
        return self.capture.get(constant)
    
    def set(self, constant, value):
        if self.current_index >= len(self.array):
            return False
        return self.capture.set(constant, value)
    
    def read(self):
        if self.current_index >= len(self.array):
            return False, None
        ret, frame = self.capture.read()
        if not ret:
            if self.current_index+1 < len(self.array):
                self.current_index += 1
                self.capture = cv2.VideoCapture(self.array[self.current_index])
                ret, frame = self.capture.read()
            else:
                False, None
        return ret, frame
    
    def release(self):
        self.capture.release()
        
    @property
    def name(self):
        if self.current_index >= len(self.array):
            return ""
        return self.names[self.current_index]


def videofiles_to_one(filenames: list, output_filename: str):
    
    def PutText(frame, word):
        font                   = cv2.FONT_HERSHEY_COMPLEX
        bottomLeftCornerOfText = (70,265)
        fontScale              = 1
        fontColor              = (255,0,0)
        thickness              = 1
        lineType               = 1
        cv2.putText(frame,word, bottomLeftCornerOfText, font, fontScale,fontColor,thickness,lineType)

    background = cv2.imread( 'background.jpg' , cv2.IMREAD_UNCHANGED)
    cap = cv2_VideoCapture_from_list(filenames)

    writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'X264'), 30, (320,280))
    #writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'MP4V'), 30, (320,280))
    #writer = cv2.VideoWriter(output_filename, 0x7634706d, 30, (320,280))

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                     )               

            comb = np.concatenate((image,background),axis=0)
            PutText(comb, cap.name)
            writer.write(comb)
            
            #cv2.imshow('Deaf Avatar', comb)
            #if cv2.waitKey(1) & 0xFF == 27:
            #    break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()


# Webservice BEGIN

class Translater:

    def __init__(self, morph=morph, the_keep_stopwords=the_keep_stopwords):
        self.morph = morph
        self.the_keep_stopwords = the_keep_stopwords

    def new_calculation(self, input_text):
        out_filenames = []
        output_text = ""
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
                out_filenames.append(stopword_filename)
                output_text += lemma.split("_")[0] + " "
                annotated_args.append(" " + word + " ")
                continue

            vocabulary_filename = vocabulary.get(lemma)
            if vocabulary_filename:
                out_filenames.append(vocabulary_filename)
                output_text += lemma.split("_")[0] + " "
                annotated_args.append(" " + word + " ")
                continue
            
            similarity_max = 0
            similarity_word = None
            for iterated_word in vocabulary.keys():
                
                if iterated_word in not_in_model:
                    continue

                if lemma in model:
                    percent = model.similarity(lemma, iterated_word)
                    if percent > similarity_max:
                        similarity_max = percent
                        similarity_word = iterated_word
            
            if similarity_word:
                filename = vocabulary.get(similarity_word)
                out_filenames.append(filename)
                output_text += similarity_word.split("_")[0] + " "
                annotated_args.append((word, similarity_word.split("_")[0] + " " + str(round(similarity_max, 2))))

        st.session_state.text_output = output_text
        st.session_state.annotations = annotated_args
        st.session_state.filenames = out_filenames
        return None

    def get_output(self):
        if "text_output" in st.session_state:
            return st.session_state.text_output
        return None

    def get_annotations(self):
        if "annotations" in st.session_state:
            return st.session_state.annotations
        return None

    def get_filenames(self):
        if "filenames" in st.session_state:
            return st.session_state.filenames
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

text = text_area.text_area(OUTPUT_LABEL, "", height=HEIGHT_TEXTAREA, disabled=True)

annotations = st.empty()


button_translate = st.button('Перевести')
if button_translate:
    
    successable = True
    start = time.time()
    with st.spinner('Вычисляем матрицы векторов...'):
        try:
            translater.new_calculation(text_input)
        except Exception as error:
            successable = False
            st.error(error)
    end = time.time()
    if successable:
        st.success('Синонимы найдены успешно. Время: {} сек.'.format(round(end - start, 3)))

    output_text = translater.get_output()
    if output_text:
        text = text_area.text_area(OUTPUT_LABEL, output_text, height=HEIGHT_TEXTAREA, disabled=True)
        st.write("\n")

    annots = translater.get_annotations()
    if annots:
        annotations = annotated_text(*annots)
        st.write("\n")
else:
    st.write('Нажмите перевести')


button_video = st.button('Запустить видео')
if button_video:

    output_text = translater.get_output()
    if output_text:
        text = text_area.text_area(OUTPUT_LABEL, output_text, height=HEIGHT_TEXTAREA, disabled=True)
        st.write("\n")

    annots = translater.get_annotations()
    if annots:
        annotations = annotated_text(*annots)
        st.write("\n")

    videofiles_list = translater.get_filenames()
    if videofiles_list:

        successable = True
        start = time.time()
        with st.spinner('Генерируем результирующее видео жестов...'):
            try:
                #clips = [VideoFileClip(c) for c in videofiles_list]
                #final_clip = concatenate_videoclips(clips)

                temp = tempfile.NamedTemporaryFile(delete=False)
                try:
                    name_temp = temp.name + ".mp4"
                finally:
                    temp.close()
                    #final_clip.write_videofile(name_temp, codec='libx264')

                    videofiles_to_one(videofiles_list, name_temp)
                    
                    video_file3 = open(name_temp, 'rb')
                    video_bytes3 = video_file3.read()
                st.video(video_bytes3, format="video/mp4")
            except Exception as error:
                successable = False
                st.error(error)
        end = time.time()
        if successable:
            st.success('Видеофайл сгенерирован успешно. Время: {} сек.'.format(round(end - start, 3)))
    else:
        st.write('Для запуска видео, сначала нажмите кнопку "Перевести"')