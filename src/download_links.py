import time
import random
import requests

from bs4 import BeautifulSoup
from util import const
from tqdm import tqdm


def main():

    search_url = 'https://search.yahoo.com/search?p={}&ei=UTF-8&b={}'

    urls = []
    try:
        for i in tqdm(range(1, 500, 10)):

            search_page = requests.get(search_url.format('Web', i))
            soup = BeautifulSoup(search_page.text, 'lxml')
            res = soup.find('div', {'id': 'res'})
            lis = res.find_all('li')
            search_results = [li.find('a')['href'] for li in lis]

            urls.extend(search_results)

            time.sleep(random.randint(3, 10))
    except Exception as err:
        print(err)
        import pdb
        pdb.set_trace()

    url_file = const.DATA_DIR / 'urls.txt'
    with url_file.open('w') as wf:
        for url in urls:
            wf.write(url + '\n')


if __name__ == '__main__':
    main()
