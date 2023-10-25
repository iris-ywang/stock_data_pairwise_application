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
import numpy as np

os.chdir('company_data/')


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


def etl_other_ratios(response, yearly_date="-12-31", col=None):

    #regex to find the data
    text=re.findall('center;\"\>(.*?)\<\/th',response.text)
    num=re.findall('center;\"\>(.*?)\<\/td',response.text)
    num = [i.replace("$","") for i in num] 

    n_col = len(text) - 1
    num_2d = [num[i:i+n_col] for i in range(0, len(num), n_col)]
    date = [num[i] for i in range(0, len(num), n_col)]

    df_list = []
    for row_num in range(len(date)):
        if yearly_date in num_2d[row_num][0]:
            df_list.append(num_2d[row_num])

    df = pd.DataFrame(df_list, columns=text[1:])

    if col is not None:
        return df.set_index(text[1])[col]
    
    return df


def single_ratio_table(ticker: str, company: str, website_subdomain: str, col_name: str, new_col_name: str, yearly_date="-12-31", col=None):
    url_pe = f'https://www.macrotrends.net/stocks/charts/{ticker}/{company}/{website_subdomain}'
    response_pe=scrape(url_pe)
    pe_ratio = etl_other_ratios(response_pe, yearly_date=yearly_date, col=col_name)
    pe_ratio.name = new_col_name
    return pe_ratio


def etl_price_change(response, yearly_date="-12-31"):
    #regex to find the data
    text=re.findall('center;\"\>(.*?)\<\/th',response.text)  # 1-table title, 2-Year, 3:7-price in year, 8-annual % change
    num=re.findall('center;\"\>(.*?)\<\/td',response.text)  # only have values for column 3:7
    num = [i.replace("$","") for i in num] 

    year=re.findall('center\"\>(.*?)\<\/td',response.text)

    n_records = int(len(num) / 5)
    recent_year = int(year[0])
    years = list(range(recent_year, recent_year-n_records, -1))

    n_col = len(text) - 3  # remove 1-table title, 2-Year, 9-annual % change
    num_2d = [num[i:i+n_col] for i in range(0, len(num), n_col)]
    date = [str(year) + yearly_date for year in years]

    df = pd.DataFrame(num_2d, columns=text[2:-1], index=date, dtype=float)
    df["annual_pc_price_change"] = df["Year Close"].pct_change(periods=-1)

    return df["annual_pc_price_change"]


def annual_pc_price_change_table(ticker: str, company: str, yearly_date="-12-31"):
    url_pc = f'https://www.macrotrends.net/stocks/charts/{ticker}/{company}/stock-price-history'
    response_pc=scrape(url_pc)
    pc_price_change = etl_price_change(response_pc, yearly_date=yearly_date)
    return pc_price_change


def download_cmopany_stock_history(ticker, company):
    print(ticker, company)
    url=f'https://www.macrotrends.net/stocks/charts/{ticker}/{company}/financial-ratios'
    response=scrape(url)
    df=etl(response)

    yearly_date = df.index[0][4:]

    pe_ratio = single_ratio_table(ticker, company, "pe-ratio", "PE Ratio", "pe-ratio", yearly_date)
    quick_ratio = single_ratio_table(ticker, company, "quick-ratio", "Quick Ratio", "quick-ratio", yearly_date)
    ps_ratio = single_ratio_table(ticker, company, "price-sales", "Price to Sales Ratio", "ps-ratio", yearly_date)
    pb_ratio = single_ratio_table(ticker, company, "price-book", "Price to Book Ratio", "pb-ratio", yearly_date)
    price_change = annual_pc_price_change_table(ticker, company, yearly_date)

    url_is = f'https://www.macrotrends.net/stocks/charts/{ticker}/{company}/income-statement'
    response_is=scrape(url_is)
    df_is=etl(response_is)
    
    df_is["operating-income"] = pd.to_numeric(df_is["operating-income"])
    df_is["net-income"] = pd.to_numeric(df_is["net-income"])
    df_is = df_is[["operating-income", "net-income"]].pct_change(periods=-1)


    df_concat = pd.concat(
        [df, df_is, pe_ratio, ps_ratio, quick_ratio, pb_ratio,], axis=1
    ).join(price_change)        
    
    columns = ["annual_pc_price_change", 
                "pe-ratio", "ps-ratio", "pb-ratio", "roe", 
                "roa", "operating-margin", "net-profit-margin",
                "debt-equity-ratio", "current-ratio", "quick-ratio", 
                "inventory-turnover", "receiveable-turnover", 
                "operating-income", "net-income"]
    df_final = pd.DataFrame(columns=columns)
    for column in columns:
        try:
            df_final[column] = df_concat[column]
        except KeyError:
            df_final[column] = np.nan
    df_final[columns].to_csv(f'{ticker}_history.csv')
    
    return


def main(tickers):
    z = list(map(str,tickers.split(',')))
    for ticker in z:
        a=ticker
        driver = webdriver.Firefox()
        url = 'https://www.macrotrends.net/'
        driver.get(url)
        box = driver.find_element(By.CSS_SELECTOR, ".js-typeahead")
        box.send_keys(a)
        time.sleep(1)
        box.send_keys(Keys.DOWN, Keys.RETURN)
        time.sleep(1)
        geturl = driver.current_url

        if "stocks" in geturl:
            geturlsp = geturl.split("/", 10)
            for split_id in range(len(geturlsp)):
                if geturlsp[split_id] == ticker:
                    company = geturlsp[split_id+1]
                    break
        driver.quit()
        download_cmopany_stock_history(ticker, company)


if __name__ == "__main__":
    # main([("MCD", "mcdonalds")])
    ticker=''
    
    main(tickers)