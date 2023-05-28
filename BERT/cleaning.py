import re
import pandas as pd

class TextCleaner:
    
    def remove_line(self, entity):
        entity = re.sub('\n','',entity) 
        return entity

    def clean_text(self, text):
        search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
        replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']
    
        # Remove tashkeel
        p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        text = re.sub(p_tashkeel,"", text)
    
        # Remove longation
        p_longation = re.compile(r'(.)\1+')
        subst = r"\1\1"
        text = re.sub(p_longation, subst, text)
    
        for i in range(0, len(search)):
            text = text.replace(search[i], replace[i])
    
        # Trim    
        text = text.strip()

        return text
    
    def re_clean(self, old_sentence, old_tags):
        space_regex = re.compile("\s+")
        new_sentence = []
        new_tags = []
        for j in range(len(old_sentence)):
            # add word if not empty and doesn't contain spaces only
            if old_sentence[j]!="" and space_regex.match(old_sentence[j])==None:
                new_sentence.append(old_sentence[j])
                new_tags.append(old_tags[j])
        return new_sentence, new_tags


class DataReader:
    
    def __init__(self, path):
        self.path = path
        
    def read_data_bert(self):
        cleaner = TextCleaner()
        
        sentences = [] 
        tags = []
        data = []

        data_txt = open(self.path,'r',encoding='utf-8')
        sentence = []
        entity = []
        
        for line in data_txt.readlines():
            if line == '\n': # Sentence end
                recleaned = cleaner.re_clean(sentence, entity)
                sentences.append(recleaned[0].copy())
                tags.append(recleaned[1].copy())
                data.append([recleaned[0].copy(),recleaned[1].copy()])
                sentence.clear()
                entity.clear()
            else:
                line = line.split(sep=' ')
                clean_word = cleaner.clean_text(line[0])       # Cleaning word
                sentence.append(clean_word)           # Add the word
                entity.append(cleaner.remove_line(line[1]))  # Clean and add entity
        print('Data:', len(data), ', Sentences:', len(sentences), ', Tags:', len(tags))        
        return data, sentences, tags