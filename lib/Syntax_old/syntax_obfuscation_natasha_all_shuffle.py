from natasha import (
    Segmenter, 
    NewsEmbedding,
    NewsSyntaxParser,
    Doc
)
from string import punctuation
import re
import random
import pymorphy2


class Syntax_natasha_all_shuffle(object):
    
    def __init__(self):
        self.segmenter = Segmenter()
        emb = NewsEmbedding()      
        self.syntax_parser = NewsSyntaxParser(emb)
        self.coordinative_conjunction = ['и', 'да', 'ни-ни', 'тоже', 'также', 'а', 'но', 'да', 'зато', 'однако', 'же' , 'или', 'либо', 'то-то']
        self.morph = pymorphy2.MorphAnalyzer()
        
    def get_tree_structure(self,sentence):
        
        doc = Doc(sentence)
        doc.segment(self.segmenter)
        doc.parse_syntax(self.syntax_parser)
        
        return doc 
    
    def text_return(self,part,tokens,depth = 1):
        text = []
        if depth == 1:
            for elem in part:
                for token in tokens:
                    if elem == token.id:
                        text.append(token.text)
                        break
        if depth == 3:
            for high_elem in part:
                for med_elem in high_elem:
                    for low_elem in med_elem:
                        for token in tokens:
                            if low_elem == token.id:
                                text.append(token.text)
                                
        return ' '.join(text)
    
    def features_extraction(self,sentence):
        
        sentence = re.sub('-с ', ' ', sentence)
        
        tokens = re.findall(r"[-—]|[\w']+|[.,!?;:]", sentence)
        
        if len(tokens) == 0:
            return sentence,False,'',False,[],False
        
        end_punct = []
        num_end_punct = 0
        for token in reversed(tokens):
            if token in punctuation:
                end_punct.append(token)
                num_end_punct = num_end_punct + 1
            else:
                break
                
        tokens =  tokens[:len(tokens) - (num_end_punct)]
        
        if len(tokens) == 0:
            return sentence,False,'',False,end_punct,False
        
        direct_speech = False
        try:
            if tokens[0] == '-':
                direct_speech = True
                tokens = tokens[1:]
        except:
            print(sentence)
        
        title = False
        if tokens[0].istitle():
             title = True
                
        for i,token in enumerate(tokens):
            parse_result = self.morph.parse(token)[0]
            if 'Name' in parse_result.tag or 'Patr' in parse_result.tag or 'Surn' in parse_result.tag:
                continue
            else:
                tokens[i] = tokens[i].lower()
                
        # Для наташи
        tokens[0] = tokens[0].title()
        
        cc = False
        cc_value = ''
        if tokens[0] in self.coordinative_conjunction:
            cc = True
            cc_value = tokens[0] 
            tokens = tokens[1:]
        
        
        
        return ' '.join(tokens),cc,cc_value,direct_speech,end_punct,title
    
    def features_insert(self,end_text,cc,cc_value,direct_speech,end_punct,title):
        # Переписать, спать хочу!
        end_text.strip()

        words = re.findall(r"[\w']+|[.,!?;]", end_text)          
        for j, word in enumerate(words):
            parse_result = self.morph.parse(word)[0]
            if 'Name' in parse_result.tag or 'Patr' in parse_result.tag or 'Surn' in parse_result.tag:
                continue
            else:
                words[j] = words[j].lower()
        end_text = ' '.join(words)

        if cc:
            end_text = cc_value.title() + ' ' + end_text
        else:
            words = re.findall(r"[\w']+|[.,!?;]", end_text) 
            for j, word in enumerate(words):
                if word in punctuation:
                    words[j] = ''
                else:
                    if title:
                        words[j] = words[j].title()
                        break 
                    else:
                        break
            end_text = ' '.join(words) 

        if direct_speech:
            end_text = '- ' + end_text
        
        for punct in end_punct:
            end_text = end_text + punct
            
        words = re.findall(r"[\w']+|[.,!?;]", end_text) 
        for j, word in enumerate(words):
            if j != len(words)-1:
                parse_result = self.morph.parse(word)[0]
                if 'PNCT' in parse_result.tag:
                    parse_result_j = self.morph.parse(words[j+1])[0]
                    if 'PNCT' in parse_result_j.tag:
                        words[j] = ''            
        end_text = ' '.join(words)
        
        return end_text
    
    def parts_extraction(self,sentence):
        doc = self.get_tree_structure(sentence)

        roots = [_.id for _ in doc.tokens if _.rel == 'root' or _.rel == 'parataxis']
        if len(roots) == 0:
            roots = [_.id for _ in doc.tokens if _.rel == 'nsubj']
        parts = []
        for root in roots:
            part = self.find_part(root,doc.tokens)    
            parts.append(self.text_return(part,doc.tokens))
        return parts
                            
    
    def find_part(self,root,tokens):     
        part = [root]
        search = True 
        while search:
            get_one = False
            for token in tokens:
                if token.head_id in part and token.id not in part:
                    get_one = True
                    part.append(token.id)
            if not get_one:
                search = False
        part.sort()
        
        return part
    
    def part_shuffler(self,sentence):
        
        doc = self.get_tree_structure(sentence)
        roots = [_.id for _ in doc.tokens if _.rel == 'root']
        if len(roots) == 0:
            roots = [_.id for _ in doc.tokens if _.rel == 'nsubj']
        shuffled_branches = []
        for root in roots:
            childrens = self.find_childrens(root,doc.tokens)
            
            branches = list(childrens.keys())

            
            random.shuffle(branches)        

            
            shuffled_branch_childrens = []
            for branch in branches:
                branch_childrens = childrens[branch]

                if len(branch_childrens) == 0:
                    shuffled_branch_childrens.append([branch]) 
                else:
                    branch_childrens.sort()
                    if max(branch_childrens) < branch:
                        shuffled_branch_childrens.append([branch] + branch_childrens)
                    elif min(branch_childrens) > branch:
                        shuffled_branch_childrens.append(branch_childrens + [branch])
                    else:
                        case = [True,False]
                        if random.choice(case):
                            shuffled_branch_childrens.append([branch] + branch_childrens)
                        else:
                            shuffled_branch_childrens.append(branch_childrens + [branch])
                    # Можно глубже нырнуть и перемешать, т.к явно больше 1 части
            # Рандомно вставляем root
            random_places = [i for i in range(0,len(shuffled_branches)+1)]
            root_place = random.choice(random_places)
            shuffled_branch_childrens.insert(root_place,[root])

            
            shuffled_branches.append(shuffled_branch_childrens)

        random.shuffle(shuffled_branches)
        
        return self.text_return(shuffled_branches,doc.tokens,3)
    
    def find_childrens(self,root,tokens):
        branches = self.find_branches(root,tokens)
        childrens = {}
        all_ = []
        for branch in branches:    
            childrens[branch] = []
        search = True 
        while search:
            get_one = False
            for token in tokens:
                if (token.head_id in branches and token.id not in all_ and token.id != root and token.head_id != root):
                    get_one = True
                    childrens[token.head_id].append(token.id)
                    all_.append(token.id)
                elif token.id not in all_ and token.id != root and token.head_id != root:
                    for branch in branches:
                        if token.head_id in childrens[branch]:
                            get_one = True
                            childrens[branch].append(token.id)
                            all_.append(token.id)
              
            if not get_one:
                search = False
        return childrens
    
    def find_branches(self,root,tokens):
        
        branches = []
        for token in tokens:
            if token.head_id == root:
                branches.append(token.id)
                
        return branches
                
        
        
    def Main_sentence_mix_natasha(self,sentence):
        sentence,cc,cc_value,direct_speech,end_punct,title = self.features_extraction(sentence)     
        parts = self.parts_extraction(sentence)
        shuffled_part = []
        for part in parts:
            part,cc_part,cc_value_part,direct_speech_part,end_punct_part,title_part= self.features_extraction(part)
            part = self.part_shuffler(part)
            part = self.features_insert(part,cc_part,cc_value_part,direct_speech_part,end_punct_part,title_part)
            
            shuffled_part.append(part)
            if part != parts[len(parts)-1]:
                shuffled_part.append(',') 
            
        shuffled_sentence = ' '.join(shuffled_part)
        
        shuffled_sentence = self.features_insert(shuffled_sentence,cc,cc_value,direct_speech,end_punct,title)
        
        return shuffled_sentence
        
        
        
        
        
    