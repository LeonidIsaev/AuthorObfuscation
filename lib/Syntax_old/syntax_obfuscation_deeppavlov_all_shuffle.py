from deeppavlov import build_model, configs
import pymorphy2
from string import punctuation
import re
import random
class Syntax_deeppavlov_all_shuffle(object):
    def __init__(self):
        self.model_deeppavlov = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=True)
        self.coordinative_conjunction = ['и', 'да', 'ни-ни', 'тоже', 'также', 'а', 'но', 'да', 'зато', 'однако', 'же' , 'или', 'либо', 'то-то']
        self.morph = pymorphy2.MorphAnalyzer()
        self.like_root = ['acl:relcl','advcl','root','parataxis','ccomp']
        self.can_be_root = ['nsubj','conj']
    
    def get_tree_structure(self,sentence):
        tree = self.model_deeppavlov([sentence])
        tree = tree[0]
        tree = re.sub('\\n','\\t',tree)
        parsed_tree = tree.split('\t')
        counter = 0
        syntax_tree = []
        tree_elems = []
        for branch in parsed_tree:
            if counter < 10:
                if branch != '_':
                    if branch.isdigit():
                        tree_elems.append(int(branch)-1)
                    else:
                        tree_elems.append(branch)
                counter = counter + 1
            else:
                syntax_tree.append(tree_elems)
                if branch.isdigit():
                    tree_elems = [int(branch)-1]
                counter = 1
        
        return syntax_tree
        

    
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
        if tokens[0] == '-':
            direct_speech = True
            tokens = tokens[1:]

        
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
    
    def check_on_root(self,tree_elems,syntax_tree):
        if tree_elems[3] in self.like_root:
            return True
        elif tree_elems[3] in self.can_be_root:
            if syntax_tree[tree_elems[2]][3] in self.like_root:
                return True
            else:
                for search_tree_elems in syntax_tree:
                    if search_tree_elems[2] == tree_elems[0] and search_tree_elems[3] == 'nsubj':
                        return True
                return False
            
    
    def get_text(self,syntax_tree):
        text = []
        for tree_elems in syntax_tree:
            text.append(tree_elems[1])
        return ' '.join(text)
    
    
    def find_parts(self,syntax_tree):
        delimiters = []
        last_delimiters = 0
        for tree_elems in syntax_tree:
            if tree_elems[3] == 'punct' or tree_elems[3] == 'cc':
                last_delimiters = tree_elems[0] 
            if self.check_on_root(tree_elems,syntax_tree):
                delimiters.append(last_delimiters)
        delimiters = [ _ for _ in delimiters if _ != 0]
        delimiters = list(set(delimiters))
        if len(delimiters) == 0:          
            return [self.get_text(syntax_tree)]
        else:
            mass_of_new_syntax_tree = []
            new_syntax_tree = []
            # Формируем новые деревья
            for i,tree_elems in enumerate(syntax_tree):
                if i not in delimiters:
                    new_syntax_tree.append(tree_elems)
                else:
                    mass_of_new_syntax_tree.append(new_syntax_tree)
                    new_syntax_tree = []
            mass_of_new_syntax_tree.append(new_syntax_tree)
            parts = []
            for elem in mass_of_new_syntax_tree:
                parts.append(self.get_text(elem))
            return parts
    
    def parts_extraction(self,sentence):
               
        syntax_tree = self.get_tree_structure(sentence)
        
        parts = self.find_parts(syntax_tree)    
        
        return parts
                            
    
    def part_shuffler(self,sentence):
        
        syntax_tree = self.get_tree_structure(sentence)
        roots = [tree_elems[0] for tree_elems in syntax_tree if tree_elems[3] == 'root']
        if len(roots) != 0:
            root = roots[0]
        else:
            return sentence
        
        childrens = self.find_childrens(root,syntax_tree)

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
        random_places = [i for i in range(0,len(shuffled_branch_childrens)+1)]
        root_place = random.choice(random_places)
        shuffled_branch_childrens.insert(root_place,[root])
        
        shuffled_part = []
        for elem in shuffled_branch_childrens:
            for data in elem:
                shuffled_part.append(self.get_text([syntax_tree[data]]))   

        return ' '.join(shuffled_part)
    
    def find_childrens(self,root,syntax_tree):
        branches = self.find_branches(root,syntax_tree)
        childrens = {}
        used = []
        for branch in branches:    
            childrens[branch] = []
        search = True 
        while search:
            get_one = False
            for tree_elems in syntax_tree:
                if (tree_elems[2] in branches and tree_elems[0] not in used and tree_elems[0] != root and tree_elems[2] != root):
                    get_one = True
                    childrens[tree_elems[2]].append(tree_elems[0])
                    used.append(tree_elems[0])
                elif tree_elems[0] not in used and tree_elems[0] != root and tree_elems[2] != root:
                    for branch in branches:
                        if tree_elems[2] in childrens[branch]:
                            get_one = True
                            childrens[branch].append(tree_elems[0])
                            used.append(tree_elems[0])             
            if not get_one:
                search = False
        return childrens
    
    def find_branches(self,root,syntax_tree):
        
        branches = []
        for tree_elems in syntax_tree:
            if tree_elems[2] == root:
                branches.append(tree_elems[0])
                
        return branches
                
        
        
    def Main_sentence_mix(self,sentence):
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
