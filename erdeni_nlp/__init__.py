def DataCleaning(text):
    import re
    cleaned_string = re.sub(re.compile('<.*>'), '', text)  #html-теги удаляем
    cleaned_string = re.sub('[^A-Za-zА-Яа-я]+', ' ', cleaned_string)
    return cleaned_string

def Preprocessing(text):
    return text.strip().lower()

def Tokenization(text):
    return text.split()

def Lemmatization(pymorphy2_MorphAnalyzer, token, keep_pos=True, convert_upos=False, 
                  drop_stopword=False, keep_stopwords: list=[]):
    p = pymorphy2_MorphAnalyzer.parse(token)[0]
    retcode = True
    content = None
    
    pymorphy2_UPOS = {
        'NOUN': 'NOUN',
        'ADJF': 'ADJ',
        'ADJS': 'ADJ',
        'COMP': 'ADJ',
        'VERB': 'VERB',
        'INFN': 'VERB',
        'PRTF': 'VERB',
        'PRTS': 'VERB',
        'GRND': 'VERB',
        'NUMR': 'NUM',
        'ADVB': 'ADV',
        'NPRO': 'PRON',
        'PRED': 'ADV',
        'PREP': 'ADP',
        'CONJ': 'SCONJ',
        'PRCL': 'PART',
        'INTJ': 'INTJ'
    }
    STOPWORDS_BY_UPOS = [ 'DET', 'SCONJ', 'PART', 'ADP', 'PRON' ]
    if drop_stopword and p.normal_form not in keep_stopwords and pymorphy2_UPOS.get(p.tag.POS, 'X') in STOPWORDS_BY_UPOS:
        retcode = False
    
    if convert_upos:
        tagging = pymorphy2_UPOS.get(p.tag.POS, 'X')
    else:
        tagging = p.tag.POS if p.tag.POS else 'X'
    
    if not keep_pos:
        content = p.normal_form
    else:
        content = p.normal_form + "_" + tagging
    
    return retcode, content

def pymorphy2_lemmas(pymorphy2_MorphAnalyzer, text="Текст нужно передать функции в виде строки!", 
                     drop_stopword=True, keep_stopwords: list=[]) -> list:
    result = []
    for word in Tokenization(Preprocessing(DataCleaning(text))):
        code, lemma = Lemmatization(pymorphy2_MorphAnalyzer, word, keep_pos=True, convert_upos=True, 
                                    drop_stopword=drop_stopword, keep_stopwords=keep_stopwords)
        if code:
            result.append(lemma)
    return result
