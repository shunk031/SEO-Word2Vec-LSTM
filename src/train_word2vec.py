import re
import gensim
from pathlib import Path
from bs4 import BeautifulSoup
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

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()
