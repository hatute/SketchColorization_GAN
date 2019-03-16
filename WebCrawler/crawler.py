from bs4 import BeautifulSoup
import os
import urllib
import urllib.request
import urllib.parse
import requests
import socket


def download_pic(url, save_path):
    succeed = True
    try:
        response = urllib.request.urlopen(url, timeout=10)
        soup = BeautifulSoup(response.read(), 'html.parser')
        download_url = soup.find('div', id='large').find('img').attrs['src']
        urllib.request.urlretrieve(download_url, save_path)
    except urllib.request.HTTPError as e:
        print('Download image ', url, ' fail')
        print('Error code: ', e.code)
        if hasattr(e, 'reason'):
            print(e.reason)
        succeed = False
    except urllib.request.URLError as ee:
        print(ee.reason)
        succeed = False
    except socket.timeout as t:
        print('Download image ', url, ' fail--timeout')
        succeed = False
    finally:
        return succeed


def download_from_page(url, page_range, save_path, start_page=1):
    total = 0
    for page in range(start_page, page_range + 1, 1):
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


def download_over_1000(url, start_id, end_id, save_path, jump=1):
    total = 0
    for id in range(start_id, end_id + 1, jump):
        name = '{:07d}.jpg'.format(id)
        succeed = download_pic(url + '/{:d}'.format(id), os.path.join(save_path, name))
        if succeed:
            total += 1
        if total % 1000 == 0:
            print('Downloaded ', total, ' images')
        if total > 80000:
            break
    print('Total download: ', total)


def login(url, data, headers):
    sess = requests.session()
    response = sess.post(url, data=data, headers=headers)
    return sess, response


if __name__ == "__main__":
    pages = 5000
    save_path = '/media/bilin/MyPassport/zerochain'
    # download_from_page('https://www.zerochan.net', pages, save_path, start_page=1001)
    download_over_1000('https://www.zerochan.net', 800000, 2000000, save_path)
