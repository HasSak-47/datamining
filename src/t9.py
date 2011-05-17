from typing import Dict, List, Tuple
from wordcloud import WordCloud
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def main():
    lines : List[str] = []
    with open('crawls.txt') as file:
        lines = [ line.replace('\n','') for line in file.readlines() ]

    common_words : List[str] = []
    with open('common.txt') as file:
        common_words = [ line.replace('\n','').lower() for line in file.readlines() ]

    starts : List[int] = []
    for idx, line in enumerate(lines):
        if 'Episode' in line:
            starts.append(idx)

    starts.append(len(lines))

    raw_crawls : List[List[str]]= []
    for start, end in zip(starts[:-1], starts[1:]):
        raw_crawls.append(lines[start:end])

    crawls : Dict[Tuple[str, str], List[str] | str] = {}
    for crawl in raw_crawls:
        crawls[(crawl[0], crawl[1])] = crawl[1:-2]

    words = defaultdict(lambda: 0)
    for crawl in crawls.values():
        for line in crawl:
            for word in line.lower().split(' '):
                if word in common_words or word == '':
                    print(word)
                    continue
                words[word] += 1


    cloud = WordCloud(
        background_color="white", min_font_size=5
    ).generate_from_frequencies(words)

    plt.close()
    plt.figure(figsize=(5, 5), facecolor=None)
    plt.imshow(cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    # plt.show()
    plt.savefig("img/words.png")
    plt.close()
