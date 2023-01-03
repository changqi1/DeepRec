#!/usr/bin/python
#****************************************************************#
# ScriptName: client.py
# Author: @alibaba-inc.com
# Create Date: 2021-05-26 10:33
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2022-12-02 17:27
# Function:
#!/usr/bin/python
#***************************************************************#

import os
import time
import requests
from multiprocessing import Pool as ThreadPool
import argparse
import threading
import random
import json


# 1024*1024
# image_url = 'https://img.alicdn.com/imgextra/O1CN01gYMjHS1DAeOTla2w9_!!0-saturn_solar.jpg'
# params={
# "image_url": image_url
# }
fname = "china_flag.txt"
with open(fname) as f:
    imageBytes = f.readline()

data = {'imageBase64': imageBytes, 'wordSize': '1'}
params = {
"input": str(data)
}

server_ip = 'http://localhost:8888'

def func(url):
  cmd = "{}{}".format(server_ip, url)
  tic = time.time()
  r = requests.post(cmd, timeout=(500, 500), data=params)
  if (r.status_code==200):
      rt = time.time() - tic
      print('request out ', r.text)
      print(rt)
      return rt
  else:
      print('error:')
      print(r.context)
      return None

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pool_size", default=16, type=int)
parser.add_argument("--max", default=200, type=int)
args, _ = parser.parse_known_args()

args.pool_size = int(os.getenv("http_process", 20))

warmup_urls = []
for i in range(args.pool_size):
  warmup_urls.append('/')

urls = []
for i in range(args.max):
  urls.append('/')
pool = ThreadPool(args.pool_size)

results = pool.map(func, warmup_urls)

tic = time.time()
results = pool.map(func, urls)
costTime = time.time()-tic  # 总耗时
rts = [r for r in results if r != None]

print()
print(">" * 15, "marvin test", ">" * 15)
print('>>>marvin: total time ', costTime, 's, qps:', len(urls) / costTime)
print('>>>marvin: avg rt:', sum(rts)/len(rts))
print("<" * 15, "marvin test", "<" * 15)
print()