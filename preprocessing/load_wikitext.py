import hashlib
import re
import sys
import tarfile
import requests
from collections import Counter, defaultdict
from pathlib import Path

fname = 'wikitext-103.tar.gz'
url = 'https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/' + fname
r = requests.get(url)
Path(f'data_raw/{fname}').write_bytes(r.content)

sha512sum = 'c8186919aa1840af6b734ea41abc580574ea8efe2fafda220f5d01002464d17566d84be5199b875136c9593f0e0678fb5d7c84bb2231de8b4151cb9c83fa2109'
sha512sum_computed = hashlib.sha512(Path('data_raw/wikitext-103.tar.gz').read_bytes()).hexdigest()
sha512sum == sha512sum_computed



# Extract the dataset
with tarfile.open(f'data_raw/{fname}') as tar:
    tar.extractall("data_raw/")

# Read train, val, and test sets into string objects
train_data = Path('data_raw/wikitext-103/wiki.train.tokens').read_text()
val_data = Path('data_raw/wikitext-103/wiki.valid.tokens').read_text()
test_data = Path('data_raw/wikitext-103/wiki.test.tokens').read_text()


# Store regular expression pattern to search for wikipedia article headings
heading_pattern = '( \n \n = [^=]*[^=] = \n \n )'

# Split out train headings and articles
train_split = re.split(heading_pattern, train_data)
train_headings = [x[7:-7] for x in train_split[1::2]]
train_articles = [x for x in train_split[2::2]]

# Split out validation headings and articles
val_split = re.split(heading_pattern, val_data)
val_headings = [x[7:-7] for x in val_split[1::2]]
val_articles = [x for x in val_split[2::2]]

# Split out test headings and articles
test_split = re.split(heading_pattern, test_data)
test_headings = [x[7:-7] for x in test_split[1::2]]
test_articles = [x for x in test_split[2::2]]

with open("data_raw/wikitext_textblock.txt", "x") as f:
    f.write(' '.join(train_articles))