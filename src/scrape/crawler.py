
# TRAINING CRAWLER

import pandas as pd
import asyncio

from src.utils.functions import scrape_urls, reshape_and_list


if __name__ == "__main__":
    # Load URLs
    df_urls = pd.read_csv('../../data/furniture_stores_pages_train.csv', index_col=0)
    df_urls.columns = ['urls']
    df_urls['domains'] = df_urls.urls.apply(lambda x: x.split('products')[0] + 'products/')

    # List of URLs to scrape and their respective domains
    urls = df_urls['urls'].tolist()
    domains = df_urls['domains'].tolist()

    # Run the scraping task
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(scrape_urls(urls, domains))
    loop.close()

    # Cleaning and saving
    df_cl = pd.DataFrame()
    df_cl['cleaned_n'] = reshape_and_list('../../data/raw_text_train.txt', 'textbreakpoint')
    df_cl['clean'] = df_cl.cleaned_n.apply(lambda x: x.replace('\n', ''))
    df_cl.to_csv('../../data/cleaned_frame_train.csv')


