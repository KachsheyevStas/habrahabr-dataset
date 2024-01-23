import nltk
import urllib.parse as urlparse
import collections
import time
import sys
import pymorphy2
import os
import json

from tqdm import tqdm
#Парсер для синтаксического разбора файлов HTML/XML
from bs4 import BeautifulSoup 

from subprocess import Popen
from nltk.corpus import stopwords
import multiprocessing as mp
from functools import partial

nltk.download('stopwords')
nltk.download('punkt')

SUBLIME_PATH = r'C:\Program Files\Sublime Text 3\sublime_text.exe'
EMED_PATH = r'C:\Program Files\EmEditor\EmEditor.exe'
DATASET_PATH = r'..\habr_posts'
FILE_BOW = r'text_test.txt'

# Морфологический анализ слов
morph = pymorphy2.MorphAnalyzer()
# Установка в качестве стоп-слов словарь русских слов
stop_words = set(stopwords.words('russian'))


# Получить пост
def get_post(post_id):
    with open(os.path.join(DATASET_PATH, str(post_id))) as fin:
        # десериализует s (экземпляр str, содержащий документ JSON) в объект Python.
        post = json.load(fin)
    return post

# Преобразовать пост        
def post_to_corpus_line(post):
    post_id = post['_id']
    author = post['author']
    tags = post['tags']
    date = post['published'][:10]

    # Позволяет нам считать количество неизменяемых объектов 
    words = collections.Counter()
    soup = BeautifulSoup(post['content_html'], 'lxml')
    [x.extract() for x in soup.findAll('code')]
    content_text = soup.getText()
    
    space_chars = u'«»“”’„*…/_.\\<>"'+u"'"
    # Избавляемся от знаков пунктуации и прочего
    for c in space_chars:
        content_text = content_text.replace(c, ' ')
    # Разбили текст на более мелкие части    
    tokens = nltk.word_tokenize(content_text)
    for token in tokens:
        if len(token) > 2:
            try:
                isnumb = float(token)
                continue
            except:
                # Заменяем ё на е
                token = token.lower().replace(u'ё', u'е')
                # Парсим токен в нормальную форму и получаем слово
                word = morph.parse(token)[0].normal_form
                # Подсче встречаемости слова в тексте
                if len(word) > 0 and word not in stop_words:
                    words[word] += 1
    # Разбор
    def parse_hub_id(hub_pair, get_name=True):
        if get_name:
            # Замена пробелов на _
            hub_id = hub_pair[0].replace(' ','_').lower()
            return hub_id
        else:
            url_parts = list(filter(lambda s: len(s) > 0,
                                    urlparse.urlsplit(hub_pair[1]).path.split('/')))
            if len(url_parts) >= 2:
                hub_id = '_'.join(url_parts[-2:])
                return hub_id
        
    hubs = []
    for hub_pair in post['hubs']:
        hub_id = parse_hub_id(hub_pair)
        if hub_id:
            # Добавление в конец списка
            hubs.append(hub_id)

    def construct_bow(words):
        return [
            (
                word.replace(' ', '_').replace(':', '_').replace('|', '_').replace('\t', '_') + 
                ('' if cnt == 1 else ':{}'.format(cnt))
            )
            for word, cnt in words.items()
        ]

    # Соединяем части: номер поста, дата публикации, автор, теги, разделы/глоб темы, текст
    parts = (
        ['{}'.format(post_id)] +
        ['|date ' + date] + 
        ['|author'] + construct_bow({author: 1} if author is not None else {}) +
        ['|tags'] + construct_bow({tag: 1 for tag in tags}) +
        ['|hubs'] + construct_bow({hub_id: 1 for hub_id in hubs}) +
        ['|words'] + construct_bow(words)
    )
    return ' '.join(parts)
# Печать времени прошедшего сначала чего-то
def print_time(cur_time, coll_size, ready_cnt):
    print(str(ready_cnt)+'/'+str(coll_size), '- {0:.0f}%'.format(100*ready_cnt/coll_size))
    print('{0:.2f} seconds'.format(cur_time),
          '({2:.0f}h{0:.0f}m{1:.0f}s) '.format(cur_time//3600,
                                               (cur_time)//60%60,
                                               cur_time-60*(cur_time//60)), end='')
    print('since begining...')
                                               
# Листинг/просмотр
def listener(q, notif_rate, coll_size):
    ready_cnt = 0
    # Отсчет времени
    start = time.time()
    with open(FILE_BOW, 'w', encoding='utf-8') as fout:
        while 1:
            # Получаем задание из очереди
            message = q.get()
            if message == 'kill':
                break
            print(message, file=fout)
            fout.flush()
            ready_cnt += 1
            if ready_cnt % (notif_rate) == 0:
                print_time(time.time()-start, coll_size, ready_cnt)
                sys.stdout.flush()
 
def worker(q, post_id):
    line = post_to_corpus_line(get_post(post_id))
    q.put(line)
    return line

def do_parallel(post_ids, notif_rate, coll_size):
    manager = mp.Manager()
    q = manager.Queue()
    func = partial(worker, q)
    # запуск нового процесса в кучу
    pool = mp.Pool(mp.cpu_count()+1)

    watcher = pool.apply_async(listener, (q, notif_rate, coll_size))
    pool.map(func, post_ids)

    q.put('kill')
    pool.close()

def do_usual(post_ids, notif_rate, coll_size):
    start = time.time()
    with open(FILE_BOW, 'w', encoding='utf-8') as fout:
        ready_cnt = 0
        for post_id in tqdm(post_ids, "Подготовка данных"):
            line = post_to_corpus_line(get_post(post_id))
            print(line, file=fout)
            ready_cnt += 1
            if ready_cnt % notif_rate == 0:
                print_time(time.time()-start, coll_size, ready_cnt)

def main(start=None, finish=None, parallel=False, notifications=20):
    post_ids = [int(filename) for filename in os.listdir(DATASET_PATH)
                if not filename.startswith('.')][start:finish]
    coll_size = len(post_ids)
    start_time = time.time()
    do_usual(post_ids, coll_size//notifications, coll_size)
    cur_time = time.time()-start_time
    
    print('\nTotal time ({0:.0f} items) from '.format(coll_size), end='')
    print('{0:.0f} - {1:.2f} seconds '.format(start, cur_time), end='')
    print('({0:.0f}h{1:.0f}m{2:.0f}s)'.format(cur_time//3600,
                                              (cur_time//60)%60,
                                              cur_time-60*(cur_time//60)))



if __name__ == '__main__':
    main(1, 760000)