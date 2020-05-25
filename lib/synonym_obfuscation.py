import pymorphy2
from string import punctuation
import gensim
import re
import random


class SynonymObfuscation(object):  
      
    def __init__(self,synonim_model_name):
        self.synonim_model_name = synonim_model_name
        if synonim_model_name == 'Ruscorpora':
            self.model_Ruscorpora = gensim.models.KeyedVectors.load_word2vec_format("D:\\AuthorObfuscation\\models\\Ruscorpora\\ruwikiruscorpora_upos_cbow_300_20_2017.bin.gz", binary=True)
            self.model_Ruscorpora.init_sims(replace=True)
        elif synonim_model_name == 'Taiga':
            self.model_Taiga = gensim.models.KeyedVectors.load("D:\\AuthorObfuscation\\models\\Taiga\\model.model")
            self.model_Taiga.init_sims(replace=True)
        self.morph = pymorphy2.MorphAnalyzer()
        self.cotags = {'ADJF' : 'ADJ',
                       'ADJS' : 'ADJ', 
                       'ADVB' : 'ADV', 
                       'COMP' : 'ADV', 
                       'GRND' : 'VERB', 
                       'INFN' : 'VERB', 
                       'NOUN' : 'NOUN', 
                       'PRED' : 'ADV', 
                       'PRTF' : 'ADJ', 
                       'PRTS' : 'VERB', 
                       'VERB' : 'VERB'}
        self.capit = re.compile('^[А-Я]+$')
        self.punct = re.compile('^(.*?)([а-яА-ЯёЁ-]+)(.*?)$')
        self.capit_letters = [chr(x) for x in range(1040,1072)] + ['Ё']
    
    def main_synonym_obfuscation(self,text):
        cash_neighb = {}
        new_text = []
        text = text.strip()
        words = text.split(' ')
        for word in words:
            struct = self.punct.findall(word)
            if struct:
                struct = struct[0]
            else:
                new_text.append(word)
                continue
            wordform = struct[1]
            if wordform:
                if self.capit.search(wordform):
                    new_text.append(word)
                    continue
                else:
                    if wordform[0] in self.capit_letters:
                        capit_flag = 1
                    else:
                        capit_flag = 0
                parse_result = self.morph.parse(wordform)[0]
                if 'Name' in parse_result.tag or 'Patr' in parse_result.tag:
                    new_text.append(word)
                    continue
                pos_flag = 0
                for tg in self.cotags:
                    if tg in parse_result.tag:
                        pos_flag = 1
                        lex = parse_result.normal_form
                        pos_tag = parse_result.tag.POS
                        if (lex, pos_tag) in cash_neighb:
                            lex_neighb = cash_neighb[(lex, pos_tag)]
                        else:
                            if pos_tag == 'NOUN':
                                gen_tag = parse_result.tag.gender
                                lex_neighb = self.search_neighbour(lex, pos_tag, gend=gen_tag)
                            else:
                                lex_neighb = self.search_neighbour(lex, pos_tag)
                            cash_neighb[(lex, pos_tag)] = lex_neighb
                        if not lex_neighb:
                            new_text.append(word)
                            break
                        else:
                            if pos_tag == 'NOUN':
                                if parse_result.tag.case == 'nomn' and parse_result.tag.number == 'sing':
                                    if capit_flag == 1:
                                        lex_neighb = lex_neighb.capitalize()
                                    new_text.append(struct[0] + lex_neighb + struct[2])
                                else:
                                    word_to_replace = self.flection(lex_neighb, parse_result.tag)
                                    if word_to_replace:
                                        if capit_flag == 1:
                                            word_to_replace = word_to_replace.capitalize()
                                        new_text.append(struct[0] + word_to_replace + struct[2])
                                    else:
                                        new_text.append(word)

                            elif pos_tag == 'ADJF':
                                if parse_result.tag.case == 'nomn' and parse_result.tag.number == 'sing':
                                    if capit_flag == 1:
                                        lex_neighb = lex_neighb.capitalize()
                                    new_text.append(struct[0] + lex_neighb + struct[2])
                                else:
                                    word_to_replace = self.flection(lex_neighb, parse_result.tag)
                                    if word_to_replace:
                                        if capit_flag == 1:
                                            word_to_replace = word_to_replace.capitalize()
                                        new_text.append(struct[0] + word_to_replace + struct[2])
                                    else:
                                        new_text.append(word)

                            elif pos_tag == 'INFN':
                                if capit_flag == 1:
                                    lex_neighb = lex_neighb.capitalize()
                                new_text.append(struct[0] + lex_neighb + struct[2])

                            elif pos_tag in ['ADVB', 'COMP', 'PRED']:
                                if capit_flag == 1:
                                    lex_neighb = lex_neighb.capitalize()
                                new_text.append(struct[0] + lex_neighb + struct[2])

                            else:
                                word_to_replace = self.flection(lex_neighb, parse_result.tag)
                                if word_to_replace:
                                    if capit_flag == 1:
                                        word_to_replace = word_to_replace.capitalize()
                                    new_text.append(struct[0] + word_to_replace + struct[2])
                                else:
                                    new_text.append(word)
                        break
                if pos_flag == 0:
                    new_text.append(word)
            else:
                new_text.append(''.join(struct))
        return ' '.join(new_text)
        
    def search_neighbour(self, word, pos, gend='masc'):
        if self.synonim_model_name == 'Ruscorpora':
            current_model = self.model_Ruscorpora
        elif self.synonim_model_name == 'Taiga':
            current_model = self.model_Taiga    
        word = word.replace('ё', 'е')
        lex = word + '_' + self.cotags[pos]
        if lex in current_model:
            neighbs = current_model.most_similar([lex], topn=20)
            find = False
            for nei in neighbs:
                if not re.findall('_',nei[0]):
                    continue 
                find = True   
                lex_n, ps_n = nei[0].split('_')
                if '::' in lex_n:
                    continue
                if self.cotags[pos] == ps_n:
                    if pos == 'NOUN':
                        parse_result = self.morph.parse(lex_n)
                        for ana in parse_result:
                            if ana.normal_form == lex_n:
                                if ana.tag.gender == gend:
                                    return lex_n
                    elif self.cotags[pos] == 'VERB' and word[-2:] == 'ся':
                        if lex_n[-2:] == 'ся':
                            return lex_n
                    elif self.cotags[pos] == 'VERB' and word[-2:] != 'ся':
                        if lex_n[-2:] != 'ся':
                            return lex_n
                    else:
                        return lex_n
            if not find:
                return neighbs[0][0]
        return None
      
    def flection(self, lex_neighb, tags):
        tags = str(tags)
        tags = re.sub(',[AGQSPMa-z-]+? ', ',', tags)
        tags = tags.replace("impf,", "")
        tags = re.sub('([A-Z]) (plur|masc|femn|neut|inan)', '\\1,\\2', tags)
        tags = tags.replace("Impe neut", "")
        tags = tags.split(',')
        tags_clean = []
        for t in tags:
            if t:
                if ' ' in t:
                    t1, t2 = t.split(' ')
                    t = t2
                tags_clean.append(t)
        tags = frozenset(tags_clean)

        prep_for_gen = self.morph.parse(lex_neighb)
        ana_array = []
        for ana in prep_for_gen:
            if ana.normal_form == lex_neighb:
                ana_array.append(ana)
        for ana in ana_array:
            try:
                flect = ana.inflect(tags)
            except:
                return None
            if flect:
                word_to_replace = flect.word
                return word_to_replace
        return None