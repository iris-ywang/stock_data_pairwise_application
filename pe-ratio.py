import re
import json
import pandas as pd
import requests
import os


os.chdir('k:/')


#simply scrape
def scrape(url,**kwargs):
    
    session=requests.Session()
    session.headers.update(
            {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'})
    
    response=session.get(url,**kwargs)

    return response



response = scrape(
    "https://www.macrotrends.net/stocks/charts/MCD/mcdonalds/financial-ratios"
)

#regex to find the data
num=re.findall('(?<=div\>\"\,)[0-9\.\"\:\-\, ]*',response.text)
text=re.findall('(?<=s\: \')\S+(?=\'\, freq)',response.text)

#convert text to dict via json
dicts=[json.loads('{'+i+'}') for i in num]
print(text)

# #create dataframe
# df=pd.DataFrame()
# for ind,val in enumerate(text):
#     df[val]=dicts[ind].values()
# df.index=dicts[ind].keys()
