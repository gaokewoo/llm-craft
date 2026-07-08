import urllib.request

from tools.log import log
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
log.info(url)
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, "the-verdict.txt")
_ = urllib.request.urlretrieve(url, file_path)

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
log.info("Total number of character: %d", len(raw_text))
log.info(raw_text[:99])
