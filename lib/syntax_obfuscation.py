from deeppavlov import build_model, configs
import pymorphy2
from string import punctuation
import re
import random
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsSyntaxParser,
    Doc
)


class SyntaxObfuscation(object):
    def __init__(self, syntax_model_name):
        self.syntax_model_name = syntax_model_name
        if syntax_model_name == 'deeppavlov':
            self.model_deeppavlov = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=True)
        elif syntax_model_name == 'natasha':
            self.segmenter = Segmenter()
            emb = NewsEmbedding()
            self.syntax_parser = NewsSyntaxParser(emb)
        else:
            print('Выберите модель, автоматически выбрана модель deeppavlov')
            self.syntax_model_name = 'deeppavlov'
            self.model_deeppavlov = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=True)

        self.conjunction = ['и', 'да', 'ни-ни', 'тоже', 'также', 'а', 'но', 'зато', 'однако', 'же',
                             'или', 'либо', 'то-то', 'что', 'будто', 'чтобы', 'чтобы не', 'как бы не', 'когда',
                             'как только', 'лишь только', 'едва', 'пока', 'в то время как', 'после того как',
                             'потому что', 'так как', 'ибо', 'оттого что', ' из-за того что', 'вследствии того что',
                             'чтобы', 'для того чтобы', 'с тем чтобы', 'если', 'если бы', 'раз', 'коль', 'коли',
                             'хотя', 'сколько ни', 'когда ни', 'что ни', 'что бы ни', 'несмотря на то что',
                             'так что', 'вследствие того что', 'как', 'будто', 'как будто', 'точно', 'словно']

        self.morph = pymorphy2.MorphAnalyzer()

        self.like_root = ['acl:relcl', 'advcl', 'root', 'parataxis', 'ccomp']

        self.can_be_root = ['nsubj', 'conj']

    def get_tree_structure(self, sentence):
        if self.syntax_model_name == 'natasha':
            doc = Doc(sentence)
            doc.segment(self.segmenter)
            doc.parse_syntax(self.syntax_parser)
            syntax_tree = {}
            for elem in doc.tokens:
                values = [elem.text, re.sub('1_', '', elem.head_id), elem.rel]
                syntax_tree[re.sub('1_', '', elem.id)] = values
        elif self.syntax_model_name == 'deeppavlov':
            tree = self.model_deeppavlov([sentence])
            tree = tree[0]
            tree = re.sub('\\n', '\\t', tree)
            parsed_tree = tree.split('\t')
            counter = 0
            syntax_tree = {}
            tree_elems = []
            for branch in parsed_tree:
                if counter < 10:
                    if branch != '_':
                        tree_elems.append(branch)
                    counter = counter + 1
                else:
                    syntax_tree[str(tree_elems[0])] = tree_elems[1:]
                    tree_elems = [branch]
                    counter = 1
        else:
            tree = self.model_deeppavlov([sentence])
            tree = tree[0]
            tree = re.sub('\\n', '\\t', tree)
            parsed_tree = tree.split('\t')
            counter = 0
            syntax_tree = {}
            tree_elems = []
            for branch in parsed_tree:
                if counter < 10:
                    if branch != '_':
                        tree_elems.append(branch)
                    counter = counter + 1
                else:
                    syntax_tree[str(tree_elems[0])] = tree_elems[1:]
                    tree_elems = [branch]
                    counter = 1

        for i, element in syntax_tree.items():
            if element[1] == '0' and element[2] != 'root':
                syntax_tree[i][2] = 'root'

        return syntax_tree

    def get_tree_structure_part(self, sentence):
        if self.syntax_model_name == 'natasha':
            doc = Doc(sentence)
            doc.segment(self.segmenter)
            doc.parse_syntax(self.syntax_parser)
            syntax_tree = {}
            for elem in doc.tokens:
                values = [elem.text, re.sub('1_', '', elem.head_id), elem.rel]
                syntax_tree[re.sub('1_', '', elem.id)] = values
        elif self.syntax_model_name == 'deeppavlov':
            tree = self.model_deeppavlov([sentence])
            tree = tree[0]
            tree = re.sub('\\n', '\\t', tree)
            parsed_tree = tree.split('\t')
            counter = 0
            syntax_tree = {}
            tree_elems = []
            for branch in parsed_tree:
                if counter < 10:
                    if branch != '_':
                        tree_elems.append(branch)
                    counter = counter + 1
                else:
                    syntax_tree[str(tree_elems[0])] = tree_elems[1:]
                    tree_elems = [branch]
                    counter = 1
        else:
            tree = self.model_deeppavlov([sentence])
            tree = tree[0]
            tree = re.sub('\\n', '\\t', tree)
            parsed_tree = tree.split('\t')
            counter = 0
            syntax_tree = {}
            tree_elems = []
            for branch in parsed_tree:
                if counter < 10:
                    if branch != '_':
                        tree_elems.append(branch)
                    counter = counter + 1
                else:
                    syntax_tree[str(tree_elems[0])] = tree_elems[1:]
                    tree_elems = [branch]
                    counter = 1

        for i, element in syntax_tree.items():
            if element[1] == '0' and element[2] != 'root':
                syntax_tree[i][2] = 'root'

        to_del = []
        with_case = []
        for i, element in syntax_tree.items():
            if element[2] == 'case':
                syntax_tree[element[1]][0] = element[0] + ' ' + syntax_tree[element[1]][0]
                with_case.append(element[1])
                to_del.append(i)
                for j, elem in syntax_tree.items():
                    if elem[2] == element[1]:
                        syntax_tree[j][2] = element[1]
            elif element[2] == 'amod' and element[1] in with_case:
                syntax_tree[element[1]][0] = syntax_tree[element[1]][0] + ' ' + element[0]
                to_del.append(i)
                for j, elem in syntax_tree.items():
                    if elem[2] == element[1]:
                        syntax_tree[j][2] = element[1]
        for elem in to_del:
            del syntax_tree[elem]

        return syntax_tree

    def features_extraction(self, sentence):

        sentence = re.sub('-с ', '', sentence)

        tokens = re.findall(r"^[-—–]|[\w]+-[\w]+|[\w']+|[.,!?;:]", sentence)

        end_punct = []
        num_end_punct = 0
        for token in reversed(tokens):
            if token in punctuation:
                end_punct.append(token)
                num_end_punct = num_end_punct + 1
            else:
                break

        tokens = tokens[:len(tokens) - (num_end_punct)]

        direct_speech = False
        if len(tokens) > 1:
            if tokens[0] == '-' or tokens[0] == '—' or tokens[0] == '–':
                direct_speech = True
                tokens = tokens[1:]

        for token in tokens:
            if token in punctuation:
                tokens = tokens[1:]
            else:
                break

        title = False
        if len(tokens) > 1:
            if tokens[0].istitle():
                title = True

        for i, token in enumerate(tokens):
            parse_result = self.morph.parse(token)[0]
            if 'Name' in parse_result.tag or 'Patr' in parse_result.tag or 'Surn' in parse_result.tag:
                continue
            else:
                tokens[i] = tokens[i].lower()


        cc_value = ''
        if len(tokens) > 1:
            if tokens[0].lower() in self.conjunction:
                cc_value = tokens[0]
                tokens = tokens[1:]
            elif ' '.join(tokens[0:2]) in self.conjunction:
                cc_value = ' '.join(tokens[0:2])
                tokens = tokens[2:]
            elif ' '.join(tokens[0:3]) in self.conjunction:
                cc_value = ' '.join(tokens[0:3])
                tokens = tokens[3:]
            elif ' '.join(tokens[0:4]) in self.conjunction:
                cc_value = ' '.join(tokens[0:4])
                tokens = tokens[4:]

        for token in tokens:
            if token in punctuation:
                tokens = tokens[1:]
            else:
                break

        if len(tokens) > 1:
            tokens[0] = tokens[0].title()

        return ' '.join(tokens), cc_value, direct_speech, end_punct, title

    def features_insert(self, end_text, cc_value, direct_speech, end_punct, title):

        end_text.strip()

        words = re.findall(r"^[-—]|[\w]+-[\w]+|[\w']+|[.,!?;:]", end_text)

        words[0] = words[0].lower()

        for j, word in enumerate(words):
            parse_result = self.morph.parse(word)[0]
            if 'Name' in parse_result.tag or 'Patr' in parse_result.tag or 'Surn' in parse_result.tag:
                continue
            else:
                words[j] = words[j].lower()
        end_text = ' '.join(words)

        if len(cc_value) > 0:
            end_text = cc_value.title() + ' ' + end_text
        else:
            words = re.findall(r"^[-—]|[\w]+-[\w]+|[\w']+|[.,!?;:]", end_text)
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

        words = re.findall(r"^[-—]|[\w]+-[\w]+|[\w']+|[.,!?;:]", end_text)
        for j, word in enumerate(words):
            if j != len(words) - 1:
                parse_result = self.morph.parse(word)[0]
                if 'PNCT' in parse_result.tag:
                    parse_result_j = self.morph.parse(words[j + 1])[0]
                    if 'PNCT' in parse_result_j.tag:
                        words[j] = ''
        end_text = ' '.join(words)

        return end_text

    def check_on_root(self, key, tree_elems, syntax_tree):
        if tree_elems[2] in self.like_root:
            return True
        elif tree_elems[2] in self.can_be_root:
            if syntax_tree[tree_elems[1]][2] in self.like_root:
                return True
            else:
                for i, search_tree_elems in syntax_tree.items():
                    if search_tree_elems[1] == key and search_tree_elems[2] == 'nsubj':
                        return True
                return False

    def get_text(self, syntax_tree):
        text = []
        for tree_elems in syntax_tree:
            text.append(tree_elems[0])
        return ' '.join(text)

    def find_parts(self, syntax_tree):
        delimiters = []
        last_delimiters = 0
        for key, tree_elems in syntax_tree.items():
            if tree_elems[2] == 'punct' or tree_elems[2] == 'cc':
                last_delimiters = key
            if self.check_on_root(key, tree_elems, syntax_tree):
                delimiters.append(last_delimiters)
        delimiters = [_ for _ in delimiters if _ != 0 and _ != '1']
        delimiters = list(set(delimiters))
        if len(delimiters) == 0:
            return [self.get_text(syntax_tree.values())]
        else:
            mass_of_new_syntax_tree = []
            new_syntax_tree = []
            # Формируем новые деревья
            for key, tree_elems in syntax_tree.items():
                if key not in delimiters:
                    new_syntax_tree.append(tree_elems)
                else:
                    mass_of_new_syntax_tree.append(new_syntax_tree)
                    new_syntax_tree = []
            mass_of_new_syntax_tree.append(new_syntax_tree)
            parts = []
            for elem in mass_of_new_syntax_tree:
                parts.append(self.get_text(elem))
            return parts

    def parts_extraction(self, sentence):

        syntax_tree = self.get_tree_structure(sentence)

        parts = self.find_parts(syntax_tree)

        return parts

    def part_shuffler_main_shuffle(self, sentence):
        syntax_tree = self.get_tree_structure_part(sentence)

        root = -1
        nsubj = -1

        for key, tree_elems in syntax_tree.items():
            if tree_elems[2] == 'root' and tree_elems[1] == '0':
                root = key
        for key, tree_elems in syntax_tree.items():
            if tree_elems[2] == 'nsubj' and tree_elems[1] == root:
                nsubj = key

        if root != -1 and nsubj != -1:
            root_childrens, nsubj_childrens = self.find_childrens_rs(root, nsubj, syntax_tree)

            branches_root = list(root_childrens.keys())
            branches_nsubj = list(nsubj_childrens.keys())

            random.shuffle(branches_root)

            shuffled_branch_childrens_root = []
            for branch in branches_root:
                branch_childrens = root_childrens[branch]
                if len(branch_childrens) == 0:
                    shuffled_branch_childrens_root.append([branch])
                else:
                    branch_childrens.sort()
                    if max(branch_childrens) < branch:
                        shuffled_branch_childrens_root.append([branch] + branch_childrens)
                    elif min(branch_childrens) > branch:
                        shuffled_branch_childrens_root.append(branch_childrens + [branch])
                    else:
                        case = [True, False]
                        if random.choice(case):
                            shuffled_branch_childrens_root.append([branch] + branch_childrens)
                        else:
                            shuffled_branch_childrens_root.append(branch_childrens + [branch])

            random_places = [i for i in range(0, len(shuffled_branch_childrens_root) + 1)]
            root_place = random.choice(random_places)
            shuffled_branch_childrens_root.insert(root_place, [root])

            random.shuffle(branches_nsubj)

            shuffled_branch_childrens_nsubj = []
            for branch in branches_nsubj:
                branch_childrens = nsubj_childrens[branch]
                if len(branch_childrens) == 0:
                    shuffled_branch_childrens_nsubj.append([branch])
                else:
                    branch_childrens.sort()
                    if max(branch_childrens) < branch:
                        shuffled_branch_childrens_nsubj.append([branch] + branch_childrens)
                    elif min(branch_childrens) > branch:
                        shuffled_branch_childrens_nsubj.append(branch_childrens + [branch])
                    else:
                        case = [True, False]
                        if random.choice(case):
                            shuffled_branch_childrens_nsubj.append([branch] + branch_childrens)
                        else:
                            shuffled_branch_childrens_nsubj.append(branch_childrens + [branch])
                    # Можно глубже нырнуть и перемешать, т.к явно больше 1 части
            # Рандомно вставляем root
            random_places = [i for i in range(0, len(shuffled_branch_childrens_nsubj) + 1)]
            nsubj_place = random.choice(random_places)
            shuffled_branch_childrens_nsubj.insert(nsubj_place, [nsubj])

            shuffled_part_root = []
            for elem in shuffled_branch_childrens_root:
                for data in elem:
                    shuffled_part_root.append(self.get_text([syntax_tree[data]]))
            shuffled_part_nsubj = []
            for elem in shuffled_branch_childrens_nsubj:
                for data in elem:
                    shuffled_part_nsubj.append(self.get_text([syntax_tree[data]]))
            if nsubj > root:
                shuffled_part = shuffled_part_nsubj + shuffled_part_root
            else:
                shuffled_part = shuffled_part_root + shuffled_part_nsubj
            return ' '.join(shuffled_part)

        elif (nsubj == -1 or root == -1) and nsubj != root:
            if root == -1:
                root = nsubj
            childrens = self.find_childrens(root, syntax_tree)

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
                        case = [True, False]
                        if random.choice(case):
                            shuffled_branch_childrens.append([branch] + branch_childrens)
                        else:
                            shuffled_branch_childrens.append(branch_childrens + [branch])
                    # Можно глубже нырнуть и перемешать, т.к явно больше 1 части
            # Рандомно вставляем root
            random_places = [i for i in range(0, len(shuffled_branch_childrens) + 1)]
            root_place = random.choice(random_places)
            shuffled_branch_childrens.insert(root_place, [root])

            shuffled_part = []
            for elem in shuffled_branch_childrens:
                for data in elem:
                    shuffled_part.append(self.get_text([syntax_tree[data]]))

            return ' '.join(shuffled_part)

        else:
            return sentence

    def part_shuffler_all_shuffle(self, sentence):
        syntax_tree = self.get_tree_structure_part(sentence)

        root = -1

        for key, tree_elems in syntax_tree.items():
            if tree_elems[2] == 'root' and tree_elems[1] == '0':
                root = key

        if root == -1:
            return sentence

        childrens = self.find_childrens(root, syntax_tree)

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
                    case = [True, False]
                    if random.choice(case):
                        shuffled_branch_childrens.append([branch] + branch_childrens)
                    else:
                        shuffled_branch_childrens.append(branch_childrens + [branch])
                    # Можно глубже нырнуть и перемешать, т.к явно больше 1 части
        # Рандомно вставляем root
        random_places = [i for i in range(0, len(shuffled_branch_childrens) + 1)]
        root_place = random.choice(random_places)
        shuffled_branch_childrens.insert(root_place, [root])

        shuffled_part = []
        for elem in shuffled_branch_childrens:
            for data in elem:
                shuffled_part.append(self.get_text([syntax_tree[data]]))

        return ' '.join(shuffled_part)

    def part_shuffler_main_reverse(self, sentence):
        syntax_tree = self.get_tree_structure_part(sentence)

        root = -1
        nsubj = -1

        for key, tree_elems in syntax_tree.items():
            if tree_elems[2] == 'root' and tree_elems[1] == '0':
                root = key
        for key, tree_elems in syntax_tree.items():
            if tree_elems[2] == 'nsubj' and tree_elems[1] == root:
                nsubj = key

        if root != -1 and nsubj != -1:
            root_childrens, nsubj_childrens = self.find_childrens_rs(root, nsubj, syntax_tree)

            branches_root = list(root_childrens.keys())
            root_childrens[root] = []
            branches_root.append(root)
            branches_nsubj = list(nsubj_childrens.keys())
            nsubj_childrens[nsubj] = []
            branches_nsubj.append(nsubj)

            branches_root.sort()

            shuffled_branch_childrens_root = []
            for branch in reversed(branches_root):
                branch_childrens = root_childrens[branch]
                if len(branch_childrens) == 0:
                    shuffled_branch_childrens_root.append([branch])
                else:
                    branch_childrens.sort()
                    if max(branch_childrens) < branch:
                        shuffled_branch_childrens_root.append([branch] + branch_childrens)
                    elif min(branch_childrens) > branch:
                        shuffled_branch_childrens_root.append(branch_childrens + [branch])
                    else:
                        case = [True, False]
                        if random.choice(case):
                            shuffled_branch_childrens_root.append([branch] + branch_childrens)
                        else:
                            shuffled_branch_childrens_root.append(branch_childrens + [branch])

            branches_nsubj.sort()

            shuffled_branch_childrens_nsubj = []
            for branch in reversed(branches_nsubj):
                branch_childrens = nsubj_childrens[branch]
                if len(branch_childrens) == 0:
                    shuffled_branch_childrens_nsubj.append([branch])
                else:
                    branch_childrens.sort()
                    if max(branch_childrens) < branch:
                        shuffled_branch_childrens_nsubj.append([branch] + branch_childrens)
                    elif min(branch_childrens) > branch:
                        shuffled_branch_childrens_nsubj.append(branch_childrens + [branch])
                    else:
                        case = [True, False]
                        if random.choice(case):
                            shuffled_branch_childrens_nsubj.append([branch] + branch_childrens)
                        else:
                            shuffled_branch_childrens_nsubj.append(branch_childrens + [branch])
                    # Можно глубже нырнуть и перемешать, т.к явно больше 1 части

            shuffled_part_root = []
            for elem in shuffled_branch_childrens_root:
                for data in elem:
                    shuffled_part_root.append(self.get_text([syntax_tree[data]]))

            shuffled_part_nsubj = []
            for elem in shuffled_branch_childrens_nsubj:
                for data in elem:
                    shuffled_part_nsubj.append(self.get_text([syntax_tree[data]]))

            if nsubj > root:
                shuffled_part = shuffled_part_nsubj + shuffled_part_root
            else:
                shuffled_part = shuffled_part_root + shuffled_part_nsubj
            return ' '.join(shuffled_part)

        elif (nsubj == -1 or root == -1) and nsubj != root:
            if root == -1:
                root = nsubj
            childrens = self.find_childrens(root, syntax_tree)

            branches = list(childrens.keys())
            childrens[root] = []
            branches.append(root)

            branches.sort()

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
                        case = [True, False]
                        if random.choice(case):
                            shuffled_branch_childrens.append([branch] + branch_childrens)
                        else:
                            shuffled_branch_childrens.append(branch_childrens + [branch])
                    # Можно глубже нырнуть и перемешать, т.к явно больше 1 части

            shuffled_part = []
            for elem in shuffled_branch_childrens:
                for data in elem:
                    shuffled_part.append(self.get_text([syntax_tree[data]]))

            return ' '.join(shuffled_part)

        else:
            return sentence

    def part_shuffler_all_reverse(self, sentence):
        syntax_tree = self.get_tree_structure_part(sentence)

        root = -1

        for key, tree_elems in syntax_tree.items():
            if tree_elems[2] == 'root' and tree_elems[1] == '0':
                root = key

        if root == -1:
            return sentence

        childrens = self.find_childrens(root, syntax_tree)

        branches = list(childrens.keys())
        childrens[root] = []
        branches.append(root)

        branches.sort()

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
                    case = [True, False]
                    if random.choice(case):
                        shuffled_branch_childrens.append([branch] + branch_childrens)
                    else:
                        shuffled_branch_childrens.append(branch_childrens + [branch])
                # Можно глубже нырнуть и перемешать, т.к явно больше 1 части

        shuffled_part = []
        for elem in shuffled_branch_childrens:
            for data in elem:
                shuffled_part.append(self.get_text([syntax_tree[data]]))

        return ' '.join(shuffled_part)

    def find_childrens_rs(self, root, nsubj, syntax_tree):
        branches_root, branches_nsubj = self.find_branches_rs(root, nsubj, syntax_tree)

        childrens_root = {}
        childrens_nsubj = {}

        used_root = []
        used_nsubj = []

        for branch in branches_root:
            childrens_root[branch] = []

        for branch in branches_nsubj:
            childrens_nsubj[branch] = []

        search = True
        while search:
            get_one = False
            for key, tree_elems in syntax_tree.items():
                if (tree_elems[1] in branches_root and key not in used_root and key != root and
                        tree_elems[1] != root):
                    get_one = True
                    childrens_root[tree_elems[1]].append(key)
                    used_root.append(key)
                elif key not in used_root and key != root and tree_elems[1] != root:
                    for branch in branches_root:
                        if tree_elems[1] in childrens_root[branch]:
                            get_one = True
                            childrens_root[branch].append(key)
                            used_root.append(key)
                elif (tree_elems[1] in branches_nsubj and key not in used_nsubj and key != nsubj and
                      tree_elems[1] != nsubj):
                    get_one = True
                    childrens_nsubj[tree_elems[1]].append(key)
                    used_nsubj.append(key)
                elif key not in used_nsubj and key != nsubj and tree_elems[1] != nsubj:
                    for branch in branches_nsubj:
                        if tree_elems[1] in childrens_nsubj[branch]:
                            get_one = True
                            childrens_nsubj[branch].append(key)
                            used_nsubj.append(key)
            if not get_one:
                search = False
        return childrens_root, childrens_nsubj

    def find_branches_rs(self, root, nsubj, syntax_tree):

        branches_root = []
        for key, tree_elems in syntax_tree.items():
            if tree_elems[1] == root and key != nsubj:
                branches_root.append(key)

        branches_nsubj = []
        for key, tree_elems in syntax_tree.items():
            if tree_elems[1] == nsubj and key != root:
                branches_nsubj.append(key)

        return branches_root, branches_nsubj

    def find_childrens(self, root, syntax_tree):
        branches = self.find_branches(root, syntax_tree)
        childrens = {}
        used = []
        for branch in branches:
            childrens[branch] = []
        search = True
        while search:
            get_one = False
            for key, tree_elems in syntax_tree.items():
                if (tree_elems[1] in branches and key not in used and key != root and tree_elems[1] != root):
                    get_one = True
                    childrens[tree_elems[1]].append(key)
                    used.append(key)
                elif key not in used and key != root and tree_elems[1] != root:
                    for branch in branches:
                        if tree_elems[1] in childrens[branch]:
                            get_one = True
                            childrens[branch].append(key)
                            used.append(key)
            if not get_one:
                search = False
        return childrens

    def find_branches(self, root, syntax_tree):

        branches = []
        for key, tree_elems in syntax_tree.items():
            if tree_elems[1] == root:
                branches.append(key)

        return branches

    def check_sentence(self, sentence):
        tokens = re.findall(r"^[-—–]|[\w]+-[\w]+|[\w']+|[.,!?;:]", sentence)
        if len(tokens) < 2:
            return False
        else:
            return True

    def main_syntax_obfuscation(self, sentence, algorithm = 'none'):
        if self.check_sentence(sentence):
            sentence, cc_value, direct_speech, end_punct, title = self.features_extraction(sentence)
            parts = self.parts_extraction(sentence)
            shuffled_part = []
            for i, part in enumerate(parts):
                part, cc_value_part, direct_speech_part, end_punct_part, title_part = self.features_extraction(part)

                if algorithm == 'all_shuffle':
                    part = self.part_shuffler_all_shuffle(part)
                elif algorithm == 'all_reverse':
                    part = self.part_shuffler_all_reverse(part)
                elif algorithm == 'main_shuffle':
                    part = self.part_shuffler_main_shuffle(part)
                elif algorithm == 'main_reverse':
                    part = self.part_shuffler_main_reverse(part)
                else:
                    print("Выберите 1 из алгоритмов: \n all_shuffle  \n all_reverse \n main_shuffle \n main_reverse")
                    return sentence + ' .'

                part = self.features_insert(part, cc_value_part, direct_speech_part, end_punct_part, title_part)
                shuffled_part.append(part)
                if i != len(parts) - 1:
                    shuffled_part.append(',')
#                 else:
#                     shuffled_part.append('.')

            shuffled_sentence = ' '.join(shuffled_part)

            shuffled_sentence = self.features_insert(shuffled_sentence, cc_value, direct_speech, end_punct, title)

            return shuffled_sentence
        else:
            return sentence + ' .'