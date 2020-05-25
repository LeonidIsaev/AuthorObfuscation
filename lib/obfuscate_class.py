from lib.syntax_obfuscation import SyntaxObfuscation
from lib.synonym_obfuscation import SynonymObfuscation
from lib.language_obfuscation import LanguageObfuscation
import pandas as pd 
import re


class Obfuscate(object):

    def __init__(self,languages, synonim_model_name, syntax_model_name, syntax_algorithm ):
        self.synonim_model_name = synonim_model_name
        self.syntax_model_name = syntax_model_name
        if self.syntax_model_name != '':
            self.syntax_obfuscator = SyntaxObfuscation(syntax_model_name)
        self.syntax_algorithm = syntax_algorithm
        self.synonim_obfuscator = SynonymObfuscation(synonim_model_name)
        self.languages = languages
        self.language_obfuscator = LanguageObfuscation(self.languages)
        
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------      
# -------------------------- Функции вспомогательные общие:----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def subdivision(self,text):
        text = re.sub('(?<=[ |\(]\w)\.(?=[^\(]*\))','',text)
        text = re.sub('«','',text)
        text = re.sub('»','',text)
        split_regex = re.compile(r'[.|!|?|…|(.*)]')
        sentences = filter(lambda t: t, [t.strip() for t in  split_regex.split(text)])
        return sentences
        
    def save_train_data(self,data,num):
        df_new = pd.DataFrame(data,columns=['author','text','obfuscated_text'])
        df_new.to_csv('D:\\AuthorObfuscation\\testdata\\obfuscated_data'+num+'.csv',index=False)
               
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
# -------------------------- Основная функкция для обфускации:-------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  

    def Main_obfuscation(self,text):

        obfuscated_sentences = []

        sentences = self.subdivision(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence == '':
                continue
            obfuscated_sentences.append(self.Obfuscate_sentence(sentence))
        return ' '.join(obfuscated_sentences)


    def Obfuscate_sentence(self,sentence):
        if self.syntax_model_name != '':
            sentence = self.syntax_obfuscator.main_syntax_obfuscation(sentence,self.syntax_algorithm)
        if self.synonim_model_name != '':
            sentence = self.synonim_obfuscator.main_synonym_obfuscation(sentence)
        if len(self.languages) != 0:
            sentence = self.language_obfuscator.main_language_obfuscation(sentence)

        return sentence + '. ' 
    
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
# -----------------МУСОР----------------------------------------------------------------------------------------------------------------------------------------------- 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

   
#     def WIKI_ENG(self,tokenize_sentence, norm_words, tags_words):
#         sentence = self.Synonim_WIKI(tokenize_sentence, norm_words, tags_words)
#         sentence = self.TranslationRU_ENG_RU(sentence)
#         return sentence
              
#     def WIKI_ENG_GER(self,tokenize_sentence, norm_words, tags_words):
#         sentence = self.Synonim_WIKI(tokenize_sentence, norm_words, tags_words)
#         sentence = self.TranslationRU_ENG_GER_RU(sentence)
#         return sentence
  
#     def OpencorporaTag_to_frozenset(self,Tag):
#         tag_list = []
#         if not Tag.animacy == None:       # одушевленность
#             tag_list.append(Tag.animacy)
#         if not Tag.aspect == None:        # вид: совершенный или несовершенный
#             tag_list.append(Tag.aspect)
#         if not Tag.case == None:          # падеж
#             tag_list.append(Tag.case)
#         if not Tag.gender == None:        # род (мужской, женский, средний)
#             tag_list.append(Tag.gender)
#         if not Tag.involvement == None:   # включенность говорящего в действие
#             tag_list.append(Tag.involvement)
#         if not Tag.mood == None:          # наклонение (повелительное, изъявительное)
#             tag_list.append(Tag.mood)
#         if not Tag.number == None:        # число (единственное, множественное)
#             tag_list.append(Tag.number)
#         if not Tag.person == None:        # лицо (1, 2, 3)
#             tag_list.append(Tag.person)
#         if not Tag.tense == None:         # время (настоящее, прошедшее, будущее)
#             tag_list.append(Tag.tense)
#         if not Tag.transitivity == None:  # переходность (переходный, непереходный)
#             tag_list.append(Tag.transitivity)
#         if not Tag.voice == None:         # залог (действительный, страдательный)
#             tag_list.append(Tag.voice)
#         return frozenset(tag_list)
        
#     def Synonim_WIKI(self,tokenize_sentence, norm_words, tags_words):     
#         new_words = {}
#         for word in norm_words.values():
#             skip = False
#             synsets = self.model_WIKI.get_synsets(word)
#             for synset in synsets:
#                 if not skip:
#                     synset.get_words()
#                     for w in synset.get_words():
#                         if not skip:
#                             if w.lemma() not in norm_words.values():
#                                 for k, v in norm_words.items():
#                                     if v == word:
#                                         new_words[k] = w.lemma()
#                                 skip = True
#                                 continue
                                
#         new_words = self.To_real_form(new_words,norm_words,tags_words)
        
#         new_text = []
#         for i,word in enumerate(tokenize_sentence):
#             if i in new_words.keys():
#                 new_text.append(new_words[i])
#             else:
#                 new_text.append(word)

#         return (' '.join(new_text))
        

#     def Synonim_My(self,tokenize_sentence, norm_words, tags_words):
#         new_words = {}
#         for word in norm_words.values():
#             skip = False
#             for k, v in norm_words.items():
#                 if v == word:
#                     tags = tags_words[k]
#             for tg in self.cotags:
#                 if tg in tags:
#                     word_tag = word+'_'+tg
#                     if word_tag in self.model_Ruscorpora:
#                         similars = self.model_Taiga.most_similar([word_tag], topn=3)
#                         if not similars == None:
#                             for similar in similars:
#                                 if not similar == None:
#                                     if not skip:
#                                         for k, v in norm_words.items():
#                                             if v == word and not skip:               
#                                                 new_words[k] = re.sub('_.*','',similar[0])
#                                                 skip = True
#                                                 continue
                                        
#         new_words = self.To_real_form(new_words,norm_words,tags_words)
        
#         new_text = []
#         for i,word in enumerate(tokenize_sentence):
#             if i in new_words.keys():
#                 new_text.append(new_words[i])
#             else:
#                 new_text.append(word)

#         return (' '.join(new_text))
        
    
        
        
#     def To_real_form(self, new_words, norm_form,tags_words):
#         for i, new_word in new_words.items():
#             norm_form = self.morph.parse(new_word)[0]
#             real_form = norm_form.inflect(self.OpencorporaTag_to_frozenset(tags_words[i]))
#             if not real_form == None:
#                 new_words[i] = real_form.word
#         return new_words   