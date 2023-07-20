import json
from transformers import RobertaTokenizerFast
import tqdm
import re

split_chars = [",", "_", "\"", ";", "[", "]", "(", ")", "{", "}", " ", "\n", "\t", ".", ":", "+", "-", "*", "/", "%", "=", "<", ">", "&", "|", "^", "!", "~", "?", "#"]
standard_lib_funcs = ['abort','abs','acos','asctime','asctime_r','asin','assert','atan','atan2','atexit','atof','atoi','atol','bsearch','btowc','calloc','catclose','catgets','catopen','ceil','clearerr','clock','cos','cosh','ctime','ctime64','ctime_r','ctime64_r','difftime','difftime64','div','erf','erfc','exit','exp','fabs','fclose','fdopen','feof','ferror','fflush','fgetc','fgetpos','fgets','fgetwc','fgetws','fileno','floor','fmod','fopen','fprintf','fputc','fputs','fputwc','fputws','fread','free','freopen','frexp','fscanf','fseek','fsetpos','ftell','fwide','fwprintf','fwrite','fwscanf','gamma','getc','getchar','getenv','gets','getwc','getwchar','gmtime','gmtime64','gmtime_r','gmtime64_r','hypot','isalnum','isalpha','isascii','isblank','iscntrl','isdigit','isgraph','islower','isprint','ispunct','isspace','isupper','iswalnum','iswalpha','iswblank','iswcntrl','iswctype','iswdigit','iswgraph','iswlower','iswprint','iswpunct','iswspace','iswupper','iswxdigit','isxdigit','j0','j1','jn','labs','ldexp','ldiv','localeconv','localtime','localtime64','localtime_r','localtime64_r','log','log10','longjmp','malloc','mblen','mbrlen','mbrtowc','mbsinit','mbsrtowcs','mbstowcs','mbtowc','memchr','memcmp','memcpy','memmove','memset','mktime','mktime64','modf','nextafter','nextafterl','nexttoward','nexttowardl','nl_langinfo','perror','pow','printf','putc','putchar','putenv','puts','putwc','putwchar','qsort','quantexpd32','quantexpd64','quantexpd128','quantized32','quantized64','quantized128','samequantumd32','samequantumd64','samequantumd128','raise','rand','rand_r','realloc','regcomp','regerror','regexec','regfree','remove','rename','rewind','scanf','setbuf','setjmp','setlocale','setvbuf','signal','sin','sinh','snprintf','sprintf','sqrt','srand','sscanf','strcasecmp','strcat','strchr','strcmp','strcoll','strcpy','strcspn','strerror','strfmon','strftime','strlen','strncasecmp','strncat','strncmp','strncpy','strpbrk','strptime','strrchr','strspn','strstr','strtod','strtod32','strtod64','strtod128','strtof','strtok','strtok_r','strtol','strtold','strtoul','strxfrm','swprintf','swscanf','system','tan','tanh','time','time64','tmpfile','tmpnam','toascii','tolower','toupper','towctrans','towlower','towupper','ungetc','ungetwc','va_arg','va_copy','va_end','va_start','vfprintf','vfscanf','vfwprintf','vfwscanf','vprintf','vscanf','vsprintf','vsnprintf','vsscanf','vswprintf','vswscanf','vwprintf','vwscanf','wcrtomb','wcscat','wcschr','wcscmp','wcscoll','wcscpy','wcscspn','wcsftime','wcslen','wcslocaleconv','wcsncat','wcsncmp','wcsncpy','wcspbrk','wcsptime','wcsrchr','wcsrtombs','wcsspn','wcsstr','wcstod','wcstod32','wcstod64','wcstod128','wcstof','wcstok','wcstol','wcstold','wcstombs','wcstoul','wcsxfrm','wctob','wctomb','wctrans','wctype','wcwidth','wmemchr','wmemcmp','wmemcpy','wmemmove','wmemset','wprintf','wscanf','y0','y1','yn']
keywords = ['uint','NULL', 'null', 'auto','break','case','char','const','continue','default','do','double','else','enum','extern','float','for','goto','if','int','long','register','return','short','signed','sizeof','static','struct','switch','typedef','union','unsigned','void','volatile','while','_Packed']
tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base", do_lower_case=True)

def rename_variables(file, new_file):
    with open(file, "r") as f:
        num_lines = sum(1 for _ in f)
    vocab = tokenizer.get_vocab()
    result = []
    bar = tqdm.tqdm(total=num_lines)
    with open(file, 'r') as f:
        for line in f:
            bar.update(1)
            js=json.loads(line.strip())
            code = js['func']
            code_old = code
            temp = code
            if False:
                for char in split_chars:
                    temp = temp.replace(char, f" ")
                words = temp.split(" ")
                words = list(set([x for x in words if x != '' and x not in standard_lib_funcs and x not in keywords]))

                words = [x for x in words if x not in vocab]
                for word in words:
                    for i in range(1, len(word) + 1):
                        if word[:i] not in vocab:
                            break
                    replacement = word[:i-1]
                    code = code.replace(word, replacement)
            code = code.replace("_", "")
            code = code.replace("\n", " ")
            #code = re.sub(r'(?!x)([a-zA-Z])([0-9]{2,})', r'\1', code)
            for i in range(20):
                code = code.replace("  ", " ")
            result.append({"func": code, "target":js['target']})

    with open(new_file, 'w') as f:
        for line in result:
            f.write(json.dumps(line)+'\n')

rename_variables('dataset/train.jsonl', 'dataset/train_renamed.jsonl')
rename_variables('dataset/test.jsonl', 'dataset/test_renamed.jsonl')
#rename_variables('dataset/valid.jsonl', 'dataset/valid_renamed.jsonl')