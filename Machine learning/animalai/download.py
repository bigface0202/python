from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

#APIの情報

key = "3edc0202289fbffacd6929f6d6403971"
secret = "4c14a21d32fd7e77"
#待ち時間（サーバーの負荷を上げない，スパムなどと認識されないように）
wait_time = 1

#保存フォルダの指定
animalname = sys.argv[1]#プログラム名の次にくる名前
savedir = "./" + animalname#animalnameというフォルダに保存する

flickr = FlickrAPI(key, secret, format = 'parsed-json')
result = flickr.photos.search(
    text = animalname,
    per_page = 400,#400枚取得する
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,#有害コンテンツは表示しない
    extras = 'url_q, licence'
)

photos = result['photos']
#返り値を表示する
# pprint(photos)

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)#url_qの場所からfilepathで保存する
    time.sleep(wait_time)