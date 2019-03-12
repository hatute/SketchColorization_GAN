from bs4 import BeautifulSoup
import os
import urllib
import urllib.request
import urllib.parse
import requests


def download_pic(url, save_path):
    response = urllib.request.urlopen(url, timeout=10)
    soup = BeautifulSoup(response.read(), 'html.parser')
    download_url = soup.find('div', id='large').find('img').attrs['src']
    urllib.request.urlretrieve(download_url, save_path)


def download_from_page(url, page_range, save_path):
    total = 0
    for page in range(1, page_range + 1, 1):
        count = 0
        response = urllib.request.urlopen(url + '/?p={:d}'.format(page), timeout=10)
        soup = BeautifulSoup(response.read(), 'html.parser')
        list = soup.find('div', id='content').find('ul', id='thumbs2').find_all('li')
        for i in list:
            img_href = i.find('a').attrs['href']
            if img_href == '/register':
                print('Member only')
                continue
            img_url = url + img_href
            name = '{:07d}.jpg'.format(int(img_href.split('/')[1]))
            download_pic(img_url, os.path.join(save_path, name))
            count += 1
        print('Download ', count, ' images from: ', url + '/?p={:d}'.format(page))
        total += count
    print('Total download: ', total)


def login(url, data, headers):
    sess = requests.session()
    response = sess.post(url, data=data, headers=headers)
    return sess, response


if __name__ == "__main__":
    pages = 5000
    save_path = ''
    download_from_page('https://www.zerochan.net', pages, save_path)
