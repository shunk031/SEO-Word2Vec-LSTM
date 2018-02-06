import requests
from tqdm import tqdm
from util import const


def main():

    url_file = const.DATA_DIR / 'urls01.txt'
    with url_file.open('r') as rf:
        urls = [line.strip() for line in rf]

    for i, url in enumerate(tqdm(urls)):
        try:
            res = requests.get(url)

            html_file = const.HTML_DIR / '{:03d}.html'.format(i)
            with html_file.open('w') as wf:
                wf.write(res.text)
        except Exception as err:
            print(err)


if __name__ == '__main__':
    main()
