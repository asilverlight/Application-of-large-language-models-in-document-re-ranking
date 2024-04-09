from transformers import LlamaTokenizer
import re
import random
import numpy as np
import copy

MAX_LEN = 2040

rng = np.random.default_rng(42)

def map_shots_name(shot):
    if shot == 0:
        return "zero"
    elif shot == 1:
        return "one"
    elif shot == 2:
        return "two"
    elif shot == 3:
        return "three"
    elif shot == 4:
        return "four"
    elif shot == 5:
        return "five"

def load_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained("/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/")
    return tokenizer

def get_length(tokenizer, text):
    ids = tokenizer.encode(text)
    return len(ids)

def get_split_num(num_all_data):
    if num_all_data > 10400:
        num_train = 10000
        num_val = 200
        num_test = 200
    elif num_all_data > 2000:
        num_test = 200
        num_val = 200
        num_train = num_all_data - num_test - num_val
    else:
        num_test = int(num_all_data * 0.1)
        num_val = num_test
        num_train = num_all_data - num_test - num_val
    return num_train, num_val, num_test

def get_train_dev_split(num_all_data):
    if num_all_data > 10200:
        num_train = 10000
        num_val = 200
    elif num_all_data > 2000:
        num_val = 200
        num_train = num_all_data - num_val
    else:
        num_val = int(num_all_data * 0.1)
        num_train = num_all_data - num_val
    return num_train, num_val

def remove_chinese_chararcter(s):
    s = s.replace("\u2018", "'")
    s = s.replace("\u2019", "'")
    s = s.replace("\u201c", "\"")
    s = s.replace("\u201d", "\"")
    s = s.replace("\u2013", "-")
    s = s.replace("\u2014", "-")
    s = s.replace("\u00a0", "")
    return s

def cap_first_char(s):
    return "".join(s[:1].upper() + s[1:])

def generate_list_prompts(list_targets, type_no):
    if type_no == 0:
        final_s = ""
        for i in range(len(list_targets)):
            s = "[" + str(i + 1) + "] " + list_targets[i]
            final_s += s + " "
    elif type_no == 1:
        final_s = ""
        for i in range(len(list_targets)):
            s = "(" + str(i + 1) + ") " + list_targets[i]
            final_s += s + " "
    elif type_no == 2:
        final_s = "\n"
        for i in range(len(list_targets)):
            s = "[" + str(i + 1) + "] " + list_targets[i]
            final_s += s + "\n"
    elif type_no == 3:
        final_s = "\n"
        for i in range(len(list_targets)):
            s = "(" + str(i + 1) + ") " + list_targets[i]
            final_s += s + "\n"
    elif type_no == 4:
        final_s = ""
        for i in range(len(list_targets)):
            s = str(i + 1) + ". " + list_targets[i]
            final_s += s + " "
    elif type_no == 5:
        final_s = "\n"
        for i in range(len(list_targets)):
            s = str(i + 1) + ". " + list_targets[i]
            final_s += s + "\n"
    return final_s

def generate_list_prompts_2(list_targets, type_no):
    if type_no == 0:
        final_s = ""
        for i in range(len(list_targets)):
            s = "[" + str(i + 1) + "] " + list_targets[i]
            final_s += s + " "
    elif type_no == 1:
        final_s = ""
        for i in range(len(list_targets)):
            s = "(" + str(i + 1) + ") " + list_targets[i]
            final_s += s + " "
    elif type_no == 2:
        final_s = ""
        for i in range(len(list_targets)):
            s = str(i + 1) + ". " + list_targets[i]
            final_s += s + " "
    return final_s.strip()

def generate_list_prompts_3(list_targets, type_no):
    if type_no == 0:
        final_s = ""
        for i in range(len(list_targets)):
            s = "[" + str(i + 1) + "] " + list_targets[i]
            final_s += s + "\n"
    elif type_no == 1:
        final_s = ""
        for i in range(len(list_targets)):
            s = "(" + str(i + 1) + ") " + list_targets[i]
            final_s += s + "\n"
    elif type_no == 2:
        final_s = ""
        for i in range(len(list_targets)):
            s = str(i + 1) + ". " + list_targets[i]
            final_s += s + "\n"
    return final_s.strip()

def find_duplicate(data):#返回1：重复；返回0：不重复
    seen = set()
    yes = True
    for item in data["pos"]:
        prefix = item["content"]
        if prefix in seen:
            yes = False
            break
        else:
            seen.add(prefix)
    if(yes):
        seen = set()
        for item in data["neg"]:
            prefix = item["content"]
            if prefix in seen:
                yes = False
                break
            else:
                seen.add(prefix)
    return not yes

def find_duplicate_db(data):
    seen = set()
    yes = True
    for item in data["doc_2"]:
        prefix = item["content"]
        if prefix in seen:
            yes = False
            break
        else:
            seen.add(prefix)
    if(yes):
        seen = set()
        for item in data["doc_1"]:
            prefix = item["content"]
            if prefix in seen:
                yes = False
                break
            else:
                seen.add(prefix)
    if(yes):
        seen = set()
        for item in data["doc_0"]:
            prefix = item["content"]
            if prefix in seen:
                yes = False
                break
            else:
                seen.add(prefix)
    return not yes

def generate_rerank_instruction(data, template, templates, yesno=True, window_size=10):
    # print(window_size)
    """
    传入一条数据,一条指令,所有指令模板,生成做好的prompt和completion
    is_exam为true:生成例子,即yes/no判断的生成1条,否则生成两条
    yesno只用于yes/no相关性判断指令生成的参数。yesno为true:生成相关,否则yesno为false:生成不相关
    其他诸如生成类的prompt,yesno变量无影响
    """
    pos_size = len(data["pos"])
    for i in range(pos_size):
        data["pos"][i]["title"] = re.sub('\n+', '\n', data["pos"][i]["title"])
        data["pos"][i]["content"] = re.sub('\n+', '\n', data["pos"][i]["content"])
        data["pos"][i]["title"] = remove_chinese_chararcter(data["pos"][i]["title"])
        data["pos"][i]["content"] = remove_chinese_chararcter(data["pos"][i]["content"])
    neg_size = len(data["neg"])
    for i in range(neg_size):
        data["neg"][i]["title"] = re.sub('\n+', '\n', data["neg"][i]["title"])
        data["neg"][i]["content"] = re.sub('\n+', '\n', data["neg"][i]["content"])
        data["neg"][i]["title"] = remove_chinese_chararcter(data["neg"][i]["title"])
        data["neg"][i]["content"] = remove_chinese_chararcter(data["neg"][i]["content"])
    
    newlist1 = []
    number0 = ''
    number1 = ''#序号
    random_num = random.choice([0, 1, 2])
    if random_num == 0:
        number0 = ''
        number1 = '.'
    if random_num == 1:
        number0 = '['
        number1 = ']'
    if random_num == 2:
        number0 = '('
        number1 = ')'
    sort_num = random.randint(0, 1)#1带大于号,0不带
    
    prompt = ""
    completion = template[1]
    neg_num = len(data["neg"])
    pos_num = len(data["pos"])
    if pos_num == 1:
        pos_item = 0
    else:
        pos_item = random.randint(0, pos_num - 1)
    if neg_num == 1:
        neg_item = 0
    else:
        neg_item = random.randint(0, neg_num - 1)
    decide = random.randint(0, 1)#0不相干,1相干,用于带有judgment的判断
    if "{query}" in template[0]:
        prompt = template[0].replace("{query}", data["query"])
    elif "{query}" in template[1]:
        completion = template[1].replace("{query}", data["query"])
    if template == templates[0] or template == templates[1]:#pointwise相关性判断
        if yesno == True:#相关
            if data["pos"][pos_item]["title"] != "":
                document = "title: {title}\ncontent: {content}".format(
                    title = data["pos"][pos_item]["title"],
                    content = data["pos"][pos_item]["content"]
                )
            else:
                document = data["pos"][pos_item]["content"]
            prompt = prompt.replace("{document}", document)
            completion = completion.replace("{judgment}", "Yes")
        else:
            if data["neg"][neg_item]["title"] != "":
                document = "title: {title}\ncontent: {content}".format(
                    title = data["neg"][neg_item]["title"],
                    content = data["neg"][neg_item]["content"]
                    )
            else:
                document = data["neg"][neg_item]["content"]
            prompt = prompt.replace("{document}", document)
            completion = completion.replace("{judgment}", "No")
    if template == templates[2] or template == templates[3]:#pointwise生成
        if data["pos"][pos_item]["title"] != "":
            document = "title: {title}\ncontent: {content}".format(
                title = data["pos"][pos_item]["title"],
                content = data["pos"][pos_item]["content"]
            )
        else:
            document = data["pos"][pos_item]["content"]
        prompt = template[0].replace("{document}", document)
    if template == templates[4] or template == templates[5]:#pairwise选择
        if decide == 0:#第一个更相关
            if data["pos"][pos_item]["title"] != "":
                document0 = "title: {title}\ncontent: {content}".format(
                    title = data["pos"][pos_item]["title"],
                    content = data["pos"][pos_item]["content"]
                )
            else:
                document0 = data["pos"][pos_item]["content"]
            if data["neg"][neg_item]["title"] != "":
                document1 = "title: {title}\ncontent: {content}".format(
                    title = data["neg"][neg_item]["title"],
                    content = data["neg"][neg_item]["content"]
                    )
            else:
                document1 = data["neg"][neg_item]["content"]
            document = f"{number0}{1}{number1} {document0}\n{number0}{2}{number1} {document1}"
            prompt = prompt.replace("{document}", document)
            if random_num == 0:
                completion = template[1].replace("{identifier}", "1")
            else:
                completion = template[1].replace("{identifier}", f"{number0}{1}{number1}")
        else:
            if data["pos"][pos_item]["title"] != "":
                document1 = "title: {title}\ncontent: {content}".format(
                    title = data["pos"][pos_item]["title"],
                    content = data["pos"][pos_item]["content"]
                )
            else:
                document1 = data["pos"][pos_item]["content"]
            if data["neg"][neg_item]["title"] != "":
                document0 = "title: {title}\ncontent: {content}".format(
                    title = data["neg"][neg_item]["title"],
                    content = data["neg"][neg_item]["content"]
                    )
            else:
                document0 = data["neg"][neg_item]["content"]
            document = f"{number0}{1}{number1} {document0}\n{number0}{2}{number1} {document1}"
            prompt = prompt.replace("{document}", document)
            if random_num == 0:
                completion = template[1].replace("{identifier}", "2")
            else:
                completion = template[1].replace("{identifier}", f"{number0}{2}{number1}")
    if template == templates[6] or template == templates[7]:#pairwise判断
        decide_order = random.randint(0, 1)#0pos-neg,1neg-pos
        
        if data["pos"][pos_item]["title"] != "":
            documentp = "title: {title}\ncontent: {content}".format(
                title = data["pos"][pos_item]["title"],
                content = data["pos"][pos_item]["content"]
            )
        else:
            documentp = data["pos"][pos_item]["content"]
        if data["neg"][neg_item]["title"] != "":
            documentn = "title: {title}\ncontent: {content}".format(
                title = data["neg"][neg_item]["title"],
                content = data["neg"][neg_item]["content"]
                )
        else:
            documentn = data["neg"][neg_item]["content"]
        if decide_order == 1:#pos-neg
            document = f"{number0}{1}{number1} {documentp}\n{number0}{2}{number1} {documentn}"
        else:#neg-pos
            document = f"{number0}{1}{number1} {documentn}\n{number0}{2}{number1} {documentp}"
        prompt = prompt.replace("{document}", document)
        if yesno == True and decide_order == 1:#yes
            prompt = prompt.replace("{num}", "1")
            completion = template[1].replace("{judgment}", "Yes")
        if yesno == True and decide_order == 0:#yes
            prompt = prompt.replace("{num}", "2")
            completion = template[1].replace("{judgment}", "Yes")
        if yesno == False and decide_order == 1:#yes
            prompt = prompt.replace("{num}", "2")
            completion = template[1].replace("{judgment}", "No")
        if yesno == False and decide_order == 0:#yes
            prompt = prompt.replace("{num}", "1")
            completion = template[1].replace("{judgment}", "No")
    if template == templates[8] or template == templates[9]:#listwise生成
        neg_doc = data["neg"]
        document = ""
        # if window_size == 10:
        #     for item in neg_doc:
        #         if item["title"]!="":
        #             document+=f"{number0}{item['neg_index']+1}{number1} title: {item['title']}\ncontent: {item['content']}\n"
        #         else:
        #             document+=f"{number0}{item['neg_index']+1}{number1} content: {item['content']}\n"
        #     prompt = prompt.replace("{document}", document)
        #     newlist = [x + 1 for x in data["relevance_order"]]
        # elif window_size != 0:
            #将多于的documents去掉，只保留window_size个
        if window_size > len(neg_doc):
            window_size = len(neg_doc)
        for i in range(window_size):
            if neg_doc[i]["title"]!="":
                document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} title: {neg_doc[i]['title']}\ncontent: {neg_doc[i]['content']}\n"
            else:
                document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} content: {neg_doc[i]['content']}\n"
        prompt = prompt.replace("{document}", document)
        newlist = [x + 1 for x in data["relevance_order"] if x < window_size]
        # else:
        #     for i in range(window_size):
        #         if neg_doc[i]["title"]!="":
        #             document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} title: {neg_doc[i]['title']}\ncontent: {neg_doc[i]['content']}\n"
        #         else:
        #             document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} content: {neg_doc[i]['content']}\n"
        #     prompt = prompt.replace("{document}", document)
        #     newlist = [x + 1 for x in data["relevance_order"] if x < window_size]
        if random_num!=0:
            if sort_num == 0:
                newlist1 = ', '.join([f'{number0}{num}{number1}' for num in newlist])
            else:
                newlist1 = ' > '.join([f'{number0}{num}{number1}' for num in newlist])
        else:
            if sort_num == 0:
                newlist1 = ', '.join([f'{num}' for num in newlist])
            else:
                newlist1 = ' > '.join([f'{num}' for num in newlist])
        completion = template[1].replace("{identifier}", newlist1)
    if template == templates[10] or template == templates[11]:#listwise判断
        if yesno == True:#判断对
            neg_doc = data["neg"]
            if window_size > len(neg_doc):
                window_size = len(neg_doc)
            document = ""
            # if window_size == 10:
            #     newlist = [x + 1 for x in data["relevance_order"]]
            #     for item in neg_doc:
            #         if item["title"]!="":
            #             document+=f"{number0}{item['neg_index']+1}{number1} title: {item['title']}\ncontent: {item['content']}\n"
            #         else:
            #             document+=f"{number0}{item['neg_index']+1}{number1} content: {item['content']}\n"
            # elif window_size != 0:
            newlist = [x + 1 for x in data["relevance_order"] if x < window_size]
            for i in range(window_size):
                if neg_doc[i]["title"]!="":
                    document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} title: {neg_doc[i]['title']}\ncontent: {neg_doc[i]['content']}\n"
                else:
                    document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} content: {neg_doc[i]['content']}\n"
            # else:
            #     newlist = [x + 1 for x in data["relevance_order"] if x < window_size]
            #     for i in range(window_size):
            #         if neg_doc[i]["title"]!="":
            #             document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} title: {neg_doc[i]['title']}\ncontent: {neg_doc[i]['content']}\n"
            #         else:
            #             document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} content: {neg_doc[i]['content']}\n"
            prompt = prompt.replace("{document}", document)
            if random_num!=0:
                if sort_num == 0:
                    newlist1 = ', '.join([f'{number0}{num}{number1}' for num in newlist])
                else:
                    newlist1 = ' > '.join([f'{number0}{num}{number1}' for num in newlist])
            else:
                if sort_num == 0:
                    newlist1 = ', '.join([f'{num}' for num in newlist])
                else:
                    newlist1 = ' > '.join([f'{num}' for num in newlist])
            prompt = prompt.replace("{identifier}", newlist1)
            completion = template[1].replace("{judgment}", "Yes")
        else:#判断错
            neg_doc = data["neg"]
            if window_size > len(neg_doc):
                window_size = len(neg_doc)
            document = ""
            # if window_size == 10:
            #     newlist = [x + 1 for x in data["relevance_order"]]
            #     for item in neg_doc:
            #         if item["title"]!="":
            #             document+=f"{number0}{item['neg_index']+1}{number1} title: {item['title']}\ncontent: {item['content']}\n"
            #         else:
            #             document+=f"{number0}{item['neg_index']+1}{number1} content: {item['content']}\n"
            # elif window_size != 0:
            newlist = [x + 1 for x in data["relevance_order"] if x < window_size]
            for i in range(window_size):
                if neg_doc[i]["title"]!="":
                    document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} title: {neg_doc[i]['title']}\ncontent: {neg_doc[i]['content']}\n"
                else:
                    document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} content: {neg_doc[i]['content']}\n"
            # else:
            #     newlist = [x + 1 for x in data["relevance_order"] if x < window_size]
            #     for i in range(window_size):
            #         if neg_doc[i]["title"]!="":
            #             document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} title: {neg_doc[i]['title']}\ncontent: {neg_doc[i]['content']}\n"
            #         else:
            #             document+=f"{number0}{neg_doc[i]['neg_index']+1}{number1} content: {neg_doc[i]['content']}\n"
            # print(window_size)
            # print(newlist)
            # print('\n')
            prompt = prompt.replace("{document}", document)
            wrong_order = list(range(1, len(newlist) + 1))
            random.shuffle(wrong_order)
            while(wrong_order == newlist):
                random.shuffle(wrong_order)#相等则重新生成
            newlist = [x for x in wrong_order]
            if random_num!=0:
                if sort_num == 0:
                    newlist1 = ', '.join([f'{number0}{num}{number1}' for num in newlist])
                else:
                    newlist1 = ' > '.join([f'{number0}{num}{number1}' for num in newlist])
            else:
                if sort_num == 0:
                    newlist1 = ', '.join([f'{num}' for num in newlist])
                else:
                    newlist1 = ' > '.join([f'{num}' for num in newlist])
            prompt = prompt.replace("{identifier}", newlist1)
            completion = template[1].replace("{judgment}", "No")
    return prompt, completion


def generate_dbpedia_instruction(data, template, templates, yesno=True):
    number0 = ''
    number1 = ''#序号
    random_num = random.choice([0, 1, 2])
    if random_num == 0:
        number0 = ''
        number1 = '.'
    if random_num == 1:
        number0 = '['
        number1 = ']'
    if random_num == 2:
        number0 = '('
        number1 = ')'
    prompt = ""
    completion = template[1]
    doc2num = len(data["doc_2"])
    for i in range(doc2num):
        data["doc_2"][i]["content"] = re.sub('\n+', '\n', data["doc_2"][i]["content"])
    doc1num = len(data["doc_1"])
    for i in range(doc1num):
        data["doc_1"][i]["content"] = re.sub('\n+', '\n', data["doc_1"][i]["content"])
    doc0num = len(data["doc_0"])
    for i in range(doc0num):
        data["doc_0"][i]["content"] = re.sub('\n+', '\n', data["doc_0"][i]["content"])
    sort_num = random.choice([0, 1])# 1带大于号,0不带
    if doc2num == 1:
        it_num2 = 0
    else:
        it_num2 = random.randint(0, doc2num - 1)
    if doc1num == 1:
        it_num1 = 0
    else:
        it_num1 = random.randint(0, doc1num - 1)
    if doc0num == 1:
        it_num0 = 0
    else:
        it_num0 = random.randint(0, doc0num - 1)
    decide = random.randint(0, 1)
    
    if "{query}" in template[0]:
        prompt = template[0].replace("{query}", data["query"])
    elif "{query}" in template[1]:
        completion = template[1].replace("{query}", data["query"])
    if template == templates[0] or template == templates[1]:#pointwise相关性判断
        if yesno == True:#相关
            if data["doc_2"][it_num2]["title"] != "":
                document = "title: {title}\ncontent: {content}".format(
                    title = data["doc_2"][it_num2]["title"],
                    content = data["doc_2"][it_num2]["content"]
                )
            else:
                document = data["doc_2"][it_num2]["content"]     
            prompt = prompt.replace("{document}", document)
            completion = completion.replace("{judgment}", "Yes")   
        else:
            if data["doc_0"][it_num0]["title"] != "":
                document = "title: {title}\ncontent: {content}".format(
                    title = data["doc_0"][it_num0]["title"],
                    content = data["doc_0"][it_num0]["content"]
                )
            else:
                document = data["doc_0"][it_num0]["content"]     
            prompt = prompt.replace("{document}", document)
            completion = completion.replace("{judgment}", "No")     
    if template == templates[2] or template == templates[3]:#pointwise生成  
        if data["doc_2"][it_num2]["title"]!="":
                document = "title: {title}\ncontent: {content}".format(
                    title = data["doc_2"][it_num2]["title"],
                    content = data["doc_2"][it_num2]["content"]
                )
        else:
            document = data["doc_2"][it_num2]["content"]
        prompt = template[0].replace("{document}", document)    
    if template == templates[4] or template == templates[5]:#pairwise选择
        if decide == 0:#第一个更相关
            if data["doc_2"][it_num2]["title"]!="":
                document0 = "title: {title}\ncontent: {content}".format(
                    title = data["doc_2"][it_num2]["title"],
                    content = data["doc_2"][it_num2]["content"]
                )
            else:
                document0 = data["doc_2"][it_num2]["content"]
            if data["doc_0"][it_num0]["title"]!="":
                document1 = "title: {title}\ncontent: {content}".format(
                    title = data["doc_0"][it_num0]["title"],
                    content = data["doc_0"][it_num0]["content"]
                )
            else:
                document1 = data["doc_0"][it_num0]["content"]
            document = f"{number0}{1}{number1} {document0}\n{number0}{2}{number1} {document1}"
            prompt = prompt.replace("{document}", document)
            if random_num == 0:
                completion = template[1].replace("{identifier}", "1")
            else:
                completion = template[1].replace("{identifier}", f"{number0}{1}{number1}")
        else:
            if data["doc_2"][it_num2]["title"]!="":
                document1 = "title: {title}\ncontent: {content}".format(
                    title = data["doc_2"][it_num2]["title"],
                    content = data["doc_2"][it_num2]["content"]
                )
            else:
                document1 = data["doc_2"][it_num2]["content"]
            if data["doc_0"][it_num0]["title"]!="":
                document0 = "title: {title}\ncontent: {content}".format(
                    title = data["doc_0"][it_num0]["title"],
                    content = data["doc_0"][it_num0]["content"]
                )
            else:
                document0 = data["doc_0"][it_num0]["content"]
            document = f"{number0}{1}{number1} {document0}\n{number0}{2}{number1} {document1}"
            prompt = prompt.replace("{document}", document)
            if random_num == 0:
                completion = template[1].replace("{identifier}", "2")
            else:
                completion = template[1].replace("{identifier}", f"{number0}{2}{number1}")          
    if template == templates[6] or template == templates[7]:#pairwise判断
        decide_order = random.randint(0, 1)
        
        if data["doc_2"][it_num2]["title"] != "":
            documentp = "title: {title}\ncontent: {content}".format(
                title = data["doc_2"][it_num2]["title"],
                content = data["doc_2"][it_num2]["content"]
            )
        else:
            documentp = data["doc_2"][it_num2]["content"]
        if data["doc_0"][it_num0]["title"] != "":
            documentn = "title: {title}\ncontent: {content}".format(
                title = data["doc_0"][it_num0]["title"],
                content = data["doc_0"][it_num0]["content"]
                )
        else:
            documentn = data["doc_0"][it_num0]["content"]
        if decide_order == 1:#pos-neg
            document = f"{number0}{1}{number1} {documentp}\n{number0}{2}{number1} {documentn}"
        else:#neg-pos
            document = f"{number0}{1}{number1} {documentn}\n{number0}{2}{number1} {documentp}"
        prompt = prompt.replace("{document}", document)
        if yesno == True and decide_order == 1:#yes
            prompt = prompt.replace("{num}", "1")
            completion = template[1].replace("{judgment}", "Yes")
        if yesno == True and decide_order == 0:#yes
            prompt = prompt.replace("{num}", "2")
            completion = template[1].replace("{judgment}", "Yes")
        if yesno == False and decide_order == 1:#yes
            prompt = prompt.replace("{num}", "2")
            completion = template[1].replace("{judgment}", "No")
        if yesno == False and decide_order == 0:#yes
            prompt = prompt.replace("{num}", "1")
            completion = template[1].replace("{judgment}", "No")                

    if data["doc_2"][it_num2]["title"]!="":
        doc2 = "title:{title}\ncontent:{content}".format(
                    title = data["doc_2"][it_num2]["title"],
                    content = data["doc_2"][it_num2]["content"]
                )
    else:
        doc2 = data["doc_2"][it_num2]["content"]
    if data["doc_1"][it_num1]["title"]!="":
        doc1 = "title:{title}\ncontent:{content}".format(
                    title = data["doc_1"][it_num1]["title"],
                    content = data["doc_1"][it_num1]["content"]
                )
    else:
        doc1 = data["doc_1"][it_num1]["content"]
    if data["doc_0"][it_num0]["title"]!="":
        doc0 = "title:{title}\ncontent:{content}".format(
                    title = data["doc_0"][it_num0]["title"],
                    content = data["doc_0"][it_num0]["content"]
                )
    else:
        doc0 = data["doc_0"][it_num0]["content"]    
    doc = dict()
    origion = [1,2,3]
    random.shuffle(origion)#shuffle之后的origin是相关性从高到低的结果
    doc[origion[0]] = doc2
    doc[origion[1]] = doc1
    doc[origion[2]] = doc0        
    origion1 = []
    document = f"{number0}{1}{number1} {doc[1]}\n{number0}{2}{number1} {doc[2]}\n{number0}{3}{number1}"
    if template == templates[8] or template == templates[9]:#listwise生成     
        if random_num!=0:
            if sort_num == 0:
                origion1 = ', '.join([f'{number0}{num}{number1}' for num in origion])
            else:
                origion1 = ' > '.join([f'{number0}{num}{number1}' for num in origion])
        else:
            if sort_num == 0:
                origion1 = ', '.join([f'{num}' for num in origion])
            else:
                origion1 = ' > '.join([f'{num}' for num in origion])
        prompt = prompt.replace("{document}", document)
        completion = template[1].replace("{identifier}", origion1)      
    if template == templates[10] or template == templates[11]:#listwise判断
        if yesno == True:#判断对
            prompt = prompt.replace("{document}", document)
            if random_num!=0:
                if sort_num == 0:
                    origion1 = ', '.join([f'{number0}{num}{number1}' for num in origion])
                else:
                    origion1 = ' > '.join([f'{number0}{num}{number1}' for num in origion])
            else:
                if sort_num == 0:
                    origion1 = ', '.join([f'{num}' for num in origion])
                else:
                    origion1 = ' > '.join([f'{num}' for num in origion])
            prompt = prompt.replace("{identifier}", origion1)
            completion = template[1].replace("{judgment}", "Yes")
        else:#判断错
            prompt = prompt.replace("{document}", document)
            wrong_order = copy.deepcopy(origion)
            random.shuffle(wrong_order)
            while(wrong_order == origion):
                random.shuffle(wrong_order)#相等则重新生成
            if random_num!=0:
                if sort_num == 0:
                    wrong_order = ', '.join([f'{number0}{num}{number1}' for num in wrong_order])
                else:
                    wrong_order = ' > '.join([f'{number0}{num}{number1}' for num in wrong_order])
            else:
                if sort_num == 0:
                    wrong_order = ', '.join([f'{num}' for num in wrong_order])
                else: 
                    wrong_order = ' > '.join([f'{num}' for num in wrong_order])
            prompt = prompt.replace("{identifier}", wrong_order)
            completion = template[1].replace("{judgment}", "No") 
    return prompt, completion


