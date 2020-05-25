from deeppavlov import build_model, configs
import pymorphy2
from string import punctuation
import re
import random
import numpy

class Syntax_deeppavlov_main_parts_reverse(object):
    def __init__(self):
        self.model_deeppavlov = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=True)
        self.coordinative_conjunction = ['и', 'да', 'ни-ни', 'тоже', 'также', 'а', 'но', 'да', 'зато', 'однако', 'же' , 'или', 'либо', 'то-то']
        self.morph = pymorphy2.MorphAnalyzer()
        self.like_root = ['acl:relcl','advcl','root','parataxis','ccomp']
        self.can_be_root = ['nsubj','conj']
        
        
    def Main_sentence_mix_deeppavlov(self,sentence):
        # Представляем предложение в виде дерева синтаксических связей
        syntax_tree = self.tree_representation_deeppavlov(sentence)
  
        end_punct = '.'
        if syntax_tree[len(syntax_tree)-1][7] == 'punct':
            end_punct = syntax_tree[len(syntax_tree)-1][1]
            syntax_tree = numpy.array(syntax_tree)
            syntax_tree = self.tree_representation_deeppavlov(' '.join(syntax_tree[:,1]))
        
        try:       
            direct_speech = False
            if syntax_tree[0][1] == '-' or syntax_tree[0][1] == '—'  :
                direct_speech = True
                ds_sent = []
                for i,tree_elems in enumerate(syntax_tree):
                    if i != 0: 
                        ds_sent.append(tree_elems[1])
                syntax_tree = self.tree_representation_deeppavlov(' '.join(ds_sent))

            cc = False
            cc_value = ''
            if syntax_tree[0][7] == 'cc':
                cc = True
                cc_value = syntax_tree[0][1]
                cc_sent = []
                for i,tree_elems in enumerate(syntax_tree):
                    if i != 0: 
                        cc_sent.append(tree_elems[1])
                syntax_tree = self.tree_representation_deeppavlov(' '.join(cc_sent))

            title = False
            if syntax_tree[0][1][0].istitle():
                 title = True
            # Считаем вложенные предложения
            parts = {}
        except:
            return sentence
        
        for i,tree_elems in enumerate(syntax_tree):
            #if (tree_elems[7] == 'root' and int(tree_elems[6]) == 0) or (tree_elems[7] == 'conj' and syntax_tree[int(tree_elems[6])-1][7] == 'root') or tree_elems[7] == 'advcl' or tree_elems[7] == 'parataxis' or tree_elems[7] == 'ccomp' or (tree_elems[7] == 'nsubj' and syntax_tree[int(tree_elems[6])-1][7] != 'root'):
            if (tree_elems[7] == 'root' and int(tree_elems[6]) == 0) or tree_elems[7] == 'advcl' or tree_elems[7] == 'parataxis' or tree_elems[7] == 'ccomp' or tree_elems[7] == 'acl:relcl' or (tree_elems[7] == 'nsubj' and syntax_tree[int(tree_elems[6])-1][7] != 'root'):      
                parts[i+1] = tree_elems[7]     
                    
        # Проверка на количество вложенных предложений                              
        if len(parts) == 1:
            end_text = self.group_swap_deeppavlov(syntax_tree)
            
            return self.clear_text_deeppavlov(end_text,cc,cc_value,direct_speech,title,end_punct)
        else:
            punct = []
            last_punct = {}
            count_punct = 0
            nopunctuation = False
            # Выделяем контрольные точки по пунктуации
            for i,tree_elems in enumerate(syntax_tree):
                if tree_elems[7] == 'cc' or tree_elems[7] == 'punct':
                    if i in last_punct.keys():
                        last_punct[i].append(tree_elems)
                        last_punct[i+1] = last_punct[i]
                    else:
                        last_punct[i+1] = [tree_elems]
                    count_punct = i + 1
                #if (tree_elems[7] == 'root' and int(tree_elems[6]) == 0) or (tree_elems[7] == 'conj' and syntax_tree[int(tree_elems[6])-1][7] == 'root') or tree_elems[7] == 'advcl' or tree_elems[7] == 'parataxis' or tree_elems[7] == 'ccomp' or (tree_elems[7] == 'nsubj' and syntax_tree[int(tree_elems[6])-1][7] != 'root'):
                if (tree_elems[7] == 'root' and int(tree_elems[6]) == 0)  or tree_elems[7] == 'advcl' or tree_elems[7] == 'parataxis' or tree_elems[7] == 'ccomp' or tree_elems[7] == 'acl:relcl' or (tree_elems[7] == 'nsubj' and syntax_tree[int(tree_elems[6])-1][7] != 'root'):
                    if int(tree_elems[0]) != list(parts.keys())[0]:
                        if count_punct not in punct and count_punct != 0 and count_punct != 1:
                            punct.append(count_punct)
                        
            mass_of_new_syntax_tree = []
            new_syntax_tree = []

            # Формируем новые деревья
            for i,tree_elems in enumerate(syntax_tree):
                if i+1 not in punct:
                    new_syntax_tree.append(tree_elems)
                else:
                    mass_of_new_syntax_tree.append(new_syntax_tree)
                    new_syntax_tree = []
            mass_of_new_syntax_tree.append(new_syntax_tree)
            texts = []

            # Обновляем представления новых деревьев
            punct_iter = 0
            for elem_of_syntax_tree in mass_of_new_syntax_tree:
                cases = []
                for tree_elems in elem_of_syntax_tree:
                    cases.append(tree_elems[1])
                short_sentences = ' '.join(cases)
                update_syntax_tree = self.tree_representation_deeppavlov(short_sentences)
                texts.append(self.group_swap_deeppavlov(update_syntax_tree))
                if punct_iter != len(punct):
                    for p in last_punct[punct[punct_iter]]:
                        texts.append(p[1])
                    punct_iter = punct_iter + 1
            
            end_text = ' '.join(texts)
            end_text.strip()
            
            words = end_text.split() 
            for j, word in enumerate(words):
                if word[len(word)-2:] == '-с':
                    words[j] = words[j][:len(word)-2]
            end_text = ' '.join(words)
            
            return self.clear_text_deeppavlov(end_text,cc,cc_value,direct_speech,title,end_punct)
    
    def get_text(self,syntax_tree):
        text = []
        for tree_elems in syntax_tree:
            text.append(tree_elems[1])
        return ' '.join(text)
    
    def clear_text_deeppavlov(self,end_text,cc,cc_value,direct_speech,title,end_punct):
        
        end_text.strip()
            
        words = end_text.split() 
        for j, word in enumerate(words):
            if word[len(word)-2:] == '-с':
                words[j] = words[j][:len(word)-2]
        end_text = ' '.join(words)

        words = end_text.split()           
        for j, word in enumerate(words):
            parse_result = self.morph.parse(word)[0]
            if 'Name' in parse_result.tag or 'Patr' in parse_result.tag or 'Surn' in parse_result.tag:
                continue
            else:
                words[j] = words[j].lower()
        end_text = ' '.join(words)

        words = end_text.split() 
        for j, word in enumerate(words):
            if j != len(words)-1:
                parse_result = self.morph.parse(word)[0]
                if 'PNCT' in parse_result.tag:
                    parse_result_j = self.morph.parse(words[j+1])[0]
                    if 'PNCT' in parse_result_j.tag:
                        words[j] = ''            
        end_text = ' '.join(words)

        if cc:
            end_text = cc_value.title() + ' ' + end_text
        else:
            words = end_text.split() 
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

        end_text = end_text +  end_punct
        
        return end_text
    
    
    def group_swap_deeppavlov(self,syntax_tree):
        nsubj = -1
        root = -1
        #Очищаем пограничные знаки препинания
        

        
        # Определяем полдлежащее и сказуемое
        for i,tree_elems in enumerate(syntax_tree):
            if tree_elems[7] == 'root' and int(tree_elems[6]) == 0:
                root = i+1
        for i,tree_elems in enumerate(syntax_tree):
            if tree_elems[7] == 'nsubj' and int(tree_elems[6]) == root :
                nsubj = i+1
                

        nsubj_group = {}
        root_group = {}

        search = True
        i_nsubj = 1
        i_root = 1
        

        if nsubj == -1 and root != -1:
            for i,tree_elems in enumerate(syntax_tree):
                if int(tree_elems[0]) == root:
                    root_group['Main'] = i+1
                if int(tree_elems[6]) == root:
                    root_group['Nearest neighbor root ' + str(i_root)] = i+1
                    i_root = i_root + 1
                
           
            while search:
                get_one = False
                for i,tree_elems in enumerate(syntax_tree):
                    if i+1 not in root_group.values():
                        i_nn = self.find_nearest_neighbor_deeppavlov(root_group,syntax_tree,i+1)
                        if i_nn != -1:
                            i_count = self.find_count_deeppavlov(i_nn,root_group,'root')
                            root_group['Nearest neighbor root groop ' + str(i_nn) + ' member ' + str(i_count)] = i+1
                            get_one = True
                            continue
                if get_one == False:
                    search = False
            if len(nsubj_group)+len(root_group) != len(syntax_tree):
                return self.get_text(syntax_tree)
                
            swapped_text = []
            parts = {}
            counter_root = 0
            for neighbor_name,neighbor in root_group.items():
                if neighbor_name.find('Nearest neighbor root groop ') == -1:
                    counter_root = counter_root + 1;
            for neighbor_name,neighbor in root_group.items():
                if neighbor_name.find('Nearest neighbor root groop ') == -1:
                    parts[counter_root] = [neighbor]
                    counter_root = counter_root - 1;
            for neighbor_name,neighbor in root_group.items(): 
                for i, part in parts.items():
                    if neighbor_name.find('Nearest neighbor root groop ' + str(part[0])) != -1:
                        parts[i].append(neighbor)
                        break
            swapped_text = []            
            for i in reversed(list(parts.keys())):
                parts[i].sort()
                for elem in parts[i]:
                    swapped_text.append(syntax_tree[elem-1][1])
            return ' '.join(swapped_text) 

        # Определяем группу полдлежащего и сказуемого
        for i,tree_elems in enumerate(syntax_tree):
            if int(tree_elems[0]) == nsubj:
                nsubj_group['Main'] = i+1
            if int(tree_elems[0]) == root:
                root_group['Main'] = i+1
            if int(tree_elems[6]) == nsubj:
                nsubj_group['Nearest neighbor nsubj ' + str(i_nsubj)] = i+1
                i_nsubj = i_nsubj + 1
            if int(tree_elems[6]) == root and tree_elems[7] != 'nsubj':
                root_group['Nearest neighbor root ' + str(i_root)] = i+1
                i_root = i_root + 1 
        while search:
            get_one = False
            if root == -1 and nsubj == -1:
                search = False
                continue             
            for i,tree_elems in enumerate(syntax_tree):
                if i+1 in nsubj_group.values() or i+1 in root_group.values():
                    continue
                if i+1 not in nsubj_group.values():
                    i_nn = self.find_nearest_neighbor_deeppavlov(nsubj_group,syntax_tree,i+1,'root')
                    if i_nn != -1:
                        i_count = self.find_count_deeppavlov(i_nn,nsubj_group,'nsubj')
                        nsubj_group['Nearest neighbor nsubj groop ' + str(i_nn) + ' member ' + str(i_count)] = i+1
                        get_one = True
                        continue
                if i+1 not in root_group.values():
                    i_nn = self.find_nearest_neighbor_deeppavlov(root_group,syntax_tree,i+1,'nsubj')
                    if i_nn != -1:
                        i_count = self.find_count_deeppavlov(i_nn,root_group,'root')
                        root_group['Nearest neighbor root groop ' + str(i_nn) + ' member ' + str(i_count)] = i+1
                        get_one = True
                        continue
            if get_one == False:
                search = False
                
        if len(nsubj_group)+len(root_group) != len(syntax_tree):
            return self.get_text(syntax_tree)
        
        
        swapped_text = []

        if nsubj < root:
            parts = {}
            counter_root = 0
            for neighbor_name,neighbor in root_group.items():
                if neighbor_name.find('Nearest neighbor root groop ') == -1:
                    counter_root = counter_root + 1;
            for neighbor_name,neighbor in root_group.items():
                if neighbor_name.find('Nearest neighbor root groop ') == -1:
                    parts[counter_root] = [neighbor]
                    counter_root = counter_root - 1;
            for neighbor_name,neighbor in root_group.items(): 
                for i, part in parts.items():
                    if neighbor_name.find('Nearest neighbor root groop ' + str(part[0])) != -1:
                        parts[i].append(neighbor)
                        break

            for i in reversed(list(parts.keys())):
                parts[i].sort()
                for elem in parts[i]:
                    swapped_text.append(syntax_tree[elem-1][1])
                    
            parts = {}
            counter_nsubj = 0
            for neighbor_name,neighbor in nsubj_group.items():
                if neighbor_name.find('Nearest neighbor nsubj groop ') == -1:
                    counter_nsubj = counter_nsubj + 1;
            for neighbor_name,neighbor in nsubj_group.items():
                if neighbor_name.find('Nearest neighbor nsubj groop ') == -1:
                    parts[counter_nsubj] = [neighbor]
                    counter_nsubj = counter_nsubj - 1;
            for neighbor_name,neighbor in nsubj_group.items():      
                for i, part in parts.items():
                    if neighbor_name.find('Nearest neighbor nsubj groop ' + str(part[0])) != -1:
                        parts[i].append(neighbor)
                        
 
            for i in reversed(list(parts.keys())):
                parts[i].sort()
                for elem in parts[i]:
                    swapped_text.append(syntax_tree[elem-1][1])
                    

        elif nsubj > root:
            parts = {}
            counter_nsubj = 0
            for neighbor_name,neighbor in nsubj_group.items():
                if neighbor_name.find('Nearest neighbor nsubj groop ') == -1:
                    counter_nsubj = counter_nsubj + 1;
            for neighbor_name,neighbor in nsubj_group.items():
                if neighbor_name.find('Nearest neighbor nsubj groop ') == -1:
                    parts[counter_nsubj] = [neighbor]
                    counter_nsubj = counter_nsubj - 1;
            for neighbor_name,neighbor in nsubj_group.items():      
                for i, part in parts.items():
                    if neighbor_name.find('Nearest neighbor nsubj groop ' + str(part[0])) != -1:
                        parts[i].append(neighbor)
                        
            for i in reversed(list(parts.keys())):
                parts[i].sort()
                for elem in parts[i]:
                    swapped_text.append(syntax_tree[elem-1][1]) 
            
            parts = {}
            counter_root = 0
            for neighbor_name,neighbor in root_group.items():
                if neighbor_name.find('Nearest neighbor root groop ') == -1:
                    counter_root = counter_root + 1;
            for neighbor_name,neighbor in root_group.items():
                if neighbor_name.find('Nearest neighbor root groop ') == -1:
                    parts[counter_root] = [neighbor]
                    counter_root = counter_root - 1;
            for neighbor_name,neighbor in root_group.items():      
                for i, part in parts.items():
                    if neighbor_name.find('Nearest neighbor root groop ' + str(part[0])) != -1:
                        parts[i].append(neighbor)
                        
            for i in reversed(list(parts.keys())):
                parts[i].sort()
                for elem in parts[i]:
                    swapped_text.append(syntax_tree[elem-1][1]) 
            
        else:
            for tree_elems in syntax_tree:
                swapped_text.append(tree_elems[2])
        return ' '.join(swapped_text) 
        


    def find_count_deeppavlov(self,i_nn,group,name):
        counter = 1
        for key in group.keys():
            if key.find('Nearest neighbor ' + name + ' groop ' + str(i_nn)) != -1:
                counter = counter + 1
        return counter
                
    
    def find_nearest_neighbor_deeppavlov(self,group,syntax_tree,index,bad_name = ''):
        searched = True

        while searched:

            for neighbor_name,neighbor in group.items():
                if bad_name != '':
                    if syntax_tree[index-1][7] == bad_name:
                        return -1
                    else:
                        if neighbor_name.find('groop') != -1:
                            continue
                        if int(syntax_tree[index-1][6]) == neighbor:
                            return neighbor
                else:
                    if neighbor_name.find('groop') != -1:
                            continue
                    if int(syntax_tree[index-1][6]) == neighbor:
                        return neighbor
            index = int(syntax_tree[index-1][6]) 
                        
        
    
    def tree_representation_deeppavlov(self,sentence):
        try:
            tree = self.model_deeppavlov([sentence])
        except:
            print(sentence)
            tree = self.model_deeppavlov([sentence])
        tree = tree[0]
        tree = re.sub('\\n','\\t',tree)
        parsed_tree = tree.split('\t')
        counter = 0
        syntax_tree = []
        tree_elems = []
        for branch in parsed_tree:
            if counter < 10:
                tree_elems.append(branch)
                counter = counter + 1
            else:
                syntax_tree.append(tree_elems)
                tree_elems = [branch]
                counter = 1
        
        return syntax_tree
    
    def punct_clearing_deeppavlov(self,syntax_tree):
        begin_group = []
        end_group = {}
        for tree_elems in reversed(syntax_tree):
            if tree_elems[7] == 'punct' and tree_elems[1] != '"':
                end_group[tree_elems[0]] = tree_elems[1]
            else:
                break
        syntax_tree = syntax_tree[:len(syntax_tree)-len(end_group)] 
        
        return syntax_tree
