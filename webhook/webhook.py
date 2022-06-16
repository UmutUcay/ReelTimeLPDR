from urllib.request import Request, urlopen
import re
import urllib
import requests
from bs4 import BeautifulSoup, SoupStrainer
import urllib.request as urllib2
import wget
import urllib
import cfscrape
import webbrowser


def replace(s, position, character):
    return s[:position] + character + s[position+1:]


#url="(http://img03.platesmania.com/22030.*/s/.*\.jpg)"
url="http://img03.platesmania.com/s"
req = Request('http://platesmania.com/tr/gallery', headers={'User-Agent': 'Mozilla/5.0'})
web_byte = urlopen(req).read()

#webpage = web_byte.decode('utf-8')
#clean_url=re.findall(url,webpage, re.MULTILINE)

soup = BeautifulSoup(web_byte, "lxml")

links = []
for link in soup.findAll('img'):
    links.append(link.get('src'))

clean_links=[]
for i in range(len(links)):
    if(links[i]!= None):
        if '.jpg' in links[i]:
            clean_links.append(links[i])

fix_list=[]

for link in clean_links:
    fix_list.append(replace(link,36,'o'))

count=0
for link in fix_list:
    if(count <10):
        webbrowser.open_new_tab(link)
        count+=1
    

