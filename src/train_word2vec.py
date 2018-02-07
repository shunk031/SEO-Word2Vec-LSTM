import re
import gensim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from bs4 import BeautifulSoup
from sklearn.manifold import TSNE
from tqdm import tqdm

from util.const import HTML_DIR


def main():

    html_files = [html_file for html_file in HTML_DIR.iterdir() if html_file.suffix == '.html']
    html_files = sorted(html_files, key=lambda x: int(x.stem))

    split_html_files = []
    for html_file in tqdm(html_files):
        with html_file.open('r') as rf:
            html = rf.read()

        soup = BeautifulSoup(html, 'lxml')

        # scriptタグとnoscriptタグを取り除く
        for s in soup.find_all('script'):
            s.extract()
        for s in soup.find_all('noscript'):
            s.extract()

        html_tag_and_element_list = list(filter(lambda x: len(x) > 0, str(soup).split('\n')))
        html_tag_and_element_list = list(map(lambda x: list(filter(lambda x: len(x) > 0,
                                                                   re.split(r'\W+', x))),
                                             html_tag_and_element_list))
        # flatten
        html_tag_and_element_list = [item for sublist in html_tag_and_element_list for item in sublist]
        split_html_files.append(' '.join(html_tag_and_element_list) + '\n')

    with (Path('.') / 'tag_and_element.txt').open('w') as wf:
        for line in split_html_files:
            wf.write(line)

    sentences = gensim.models.word2vec.LineSentence('tag_and_element.txt')
    model = gensim.models.Word2Vec(sentences)

    vocab = model.wv.vocab
    embed_list = [model[v] for v in vocab]
    X_word2vec = np.vstack(embed_list)

    tsne = TSNE(n_components=2, random_state=0)
    tsne.fit_transform(X_word2vec)

    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(111)

    # 上位100単語をプロット
    embedding_x = tsne.embedding_[0:100, 0]
    embedding_y = tsne.embedding_[0:100, 1]
    ax.scatter(embedding_x, embedding_y)

    for i, (label, x, y) in enumerate(zip(vocab, embedding_x, embedding_y)):
        ax.annotate(label, xy=(x, y), xytext=(0, 0), textcorrds='offset points')

        if i == 100:
            break

    plt.savefig('test.png')


if __name__ == '__main__':
    main()
