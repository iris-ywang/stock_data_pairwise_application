# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:49:09 2021

@author: Administrator
"""


#this website is called macrotrends
#this script is designed to scrape its financial statements
#yahoo finance only contains the recent 5 year
#macrotrends can trace back to 2005 if applicable
import re
import json
import pandas as pd
import requests
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.keys import Keys

os.chdir('k:/')


#simply scrape
def scrape(url,**kwargs):
    
    session=requests.Session()
    session.headers.update(
            {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'})
    
    response=session.get(url,**kwargs)

    return response


#create dataframe
def etl(response):

    #regex to find the data
    num=re.findall('(?<=div\>\"\,)[0-9\.\"\:\-\, ]*',response.text)
    text=re.findall('(?<=s\: \')\S+(?=\'\, freq)',response.text)

    #convert text to dict via json
    dicts=[json.loads('{'+i+'}') for i in num]

    #create dataframe
    df=pd.DataFrame()
    for ind,val in enumerate(text):
        df[val]=dicts[ind].values()
    df.index=dicts[ind].keys()
    
    return df


def main(ticker_company_list):
    # missing_columns = ["pe-ratio", "quick-ratio", "income-statement/operational-income", "income-statement/net-income"]
    for ticker, company in ticker_company_list:
        # url_is = f'https://www.macrotrends.net/stocks/charts/{ticker}/{company}/income-statement'
        # response_is=scrape(url_is)
        # df_is=etl(response_is)
       
        # df_is["operating-income"] = pd.to_numeric(df_is["operating-income"])
        # df_is["net-income"] = pd.to_numeric(df_is["net-income"])

        # print(df_is[["operating-income", "net-income"]].dtypes)
        # df_is = df_is[["operating-income", "net-income"]].pct_change(periods=-1)


        url=f'https://www.macrotrends.net/stocks/charts/{ticker}/{company}/stock-price-history'
        response=scrape(url)
        df=etl(response)
        # df = pd.concat([df, df_is], axis=1)

        # url_pe = f'https://www.macrotrends.net/stocks/charts/{ticker}/{company}/microsoft/financial-ratios?freq=A'
        df.to_csv(f'{ticker}_price.csv')
    
    return


if __name__ == "__main__":
    main([("MCD", "mcdonalds")])
    # tickers = input('Tickers (separated by ",": ')
    # z = list(map(str,tickers.split(',')))
    # for i in z:
    #     a=i
    #     print(a)
    #     driver = webdriver.Firefox()
    #     url = 'https://www.macrotrends.net/'
    #     driver.get(url)
    #     box = driver.find_element(By.CSS_SELECTOR, ".js-typeahead")
    #     box.send_keys(a)
    #     time.sleep(1)
    #     box.send_keys(Keys.DOWN, Keys.RETURN)
    #     time.sleep(1)
    #     geturl = driver.current_url

    #     if "stocks" in geturl:
    #         geturlsp = geturl.split("/", 10)
    #         geturlf = url+"stocks/charts/"+geturlsp[5]+"/"+geturlsp[6]+"/"
    #         driver = webdriver.Firefox()
    #         fsurl = geturlf+"financial-statements"
    #         print(fsurl)