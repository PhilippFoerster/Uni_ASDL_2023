import subprocess
import json
from xml.etree import ElementTree as ET
import tqdm

def data2ast(file, new_file):
    with open(file, "r") as f:
        num_lines = sum(1 for _ in f)
    trees_flattened = []
    bar = tqdm.tqdm(total=num_lines)
    with open(file, 'r') as f:
        for line in f:
            bar.update(1)
            js=json.loads(line.strip())
            code = js['func']

            #write code to temp file because escaping the possible quotes in code doesn't work
            with open('temp.c', 'w') as f:
                f.write(code)
                
            command = f'"c:/program files/srcml/srcml.exe" temp.c'
            xml = subprocess.getoutput(command)
            tree = ET.fromstring(xml)
            flattened = []
            for elem in tree.iter():
                name = elem.tag.split('}')[-1]
                text = elem.text
                #print(name, text)
                flattened.append(f'<{name}>')
                if text:
                    flattened.append(text)
            trees_flattened.append({"func": " ".join(flattened), "target":js['target']})

    with open(new_file, 'w') as f:
        for line in trees_flattened:
            f.write(json.dumps(line)+'\n')
            
def remove_tokens(file, new_file):
    tokens_to_remove = ['<unit>', '<block>', '<block_content>', '<modifier>', '<function>', '<name>', '<if_stmt>', '<if>', '<else>', '<operator>', '<return>', '<break>', 
                        "<case>", "<continue>", "<do>", "<for>", "<goto>", "<switch>", "<while>", "<goto>", "<sizeof>", "<literal>", "<value>", "<expr>", "<ifdef>", "<endif>"]
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(new_file, 'w') as f:
        for line in lines:
            for token in tokens_to_remove:
                line = line.replace(token, '')
            f.write(line)

remove_tokens('dataset/train_ast2.jsonl', 'dataset/train_ast_no_tokens.jsonl')
remove_tokens('dataset/test_ast2.jsonl', 'dataset/test_ast_no_tokens.jsonl')
#data2ast('dataset/train.jsonl', 'dataset/train_ast.jsonl')
#data2ast('dataset/test.jsonl', 'dataset/test_ast.jsonl')


