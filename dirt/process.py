# coding:utf-8
import sys
import json
import re
import logging

logging.basicConfig(level=logging.INFO)
for line in sys.stdin:
    match = ''
    line = line.strip()
    if not line:
        continue
    items = line.split('\t')
    if len(items) != 2:
        continue
    if items[1] != '[]':
        continue
    else:
        # info = json.loads(items[1])
        # match =[' : '.join(_) for _ in info]
        match = items[1]
    data = json.loads(items[0], encoding='utf-8')
    logging.info(items[1])
    # match = json.loads(items[1], encoding='utf-8')
    # if match:
    #     continue
    elems = [i.strip() for i in data.get('text').split('|||')]
    gid = str(data.get('gid') or '')
    if not gid:
        continue
    if len(elems) != 3:
        continue
    title = re.sub('title:', '', elems[0])
    content = re.sub("content:", "", elems[1])
    ocr = re.sub("ocr:", "", elems[2])
    url = 'https://sg-content.bytedance.net/detail/#/?item_type=auto&item_id=' + gid
    print('\t'.join([gid, url, title, ocr, match]))

'''关键词召回2805/7069.=40%'''
