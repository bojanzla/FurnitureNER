
# MAIN UTILITIES

import asyncio
import aiohttp
import bs4
import pandas as pd
from typing import List, Tuple
from urllib.parse import urljoin
import re
from unidecode import unidecode
from keras.models import Model
from numpy import dot
from numpy.linalg import norm


def clean(html: str) -> str:
    """
    Cleans HTML and extracts text.

    Args:
        html (str): The HTML content to be cleaned.

    Returns:
        str: The cleaned text.

    Raises:
        None

    """
    if html is None:
        return ''
    soup = bs4.BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='textbreakpoint')
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    """
    Fetches the response from a given URL asynchronously.

    Args:
        session (aiohttp.ClientSession): The aiohttp client session.
        url (str): The URL to fetch.

    Returns:
        str: The response text.

    Raises:
        None

    """
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text()
            else:
                print(f"Error fetching {url}: {response.status}")
    except aiohttp.ClientError as e:
        print(f"Error fetching {url}: {str(e)}")
    except asyncio.TimeoutError:
        print(f"Timeout fetching {url}")
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")


def check_substring(strings: List[str], target_string: str) -> bool:
    """
    Checks if any substring from the given list of strings is present in the target string.

    Args:
        strings (List[str]): The list of strings to check as substrings.
        target_string (str): The target string to search in.

    Returns:
        bool: True if any substring is present, False otherwise.

    Raises:
        None

    """
    for string in strings:
        if target_string.startswith(string):
            return True
    return False


async def crawl(session: aiohttp.ClientSession, url: str, domains: List[str], full_domain: bool) -> Tuple[
    str, List[str]]:
    """
    Crawls a given URL and extracts the response and secondary URLs.

    Args:
        session (aiohttp.ClientSession): The aiohttp client session.
        url (str): The URL to crawl.
        domains (List[str]): The list of domain substrings to match against the URL.
        full_domain (bool): Indicates whether to include the full domain in the extraction.

    Returns:
        Tuple[str, List[str]]: A tuple containing the response text and the list of secondary URLs.

    Raises:
        None

    """
    secondary_urls = []
    if check_substring(domains, url):
        print(url)
        try:
            response = await fetch(session, url)
            if full_domain:
                soup = bs4.BeautifulSoup(response, 'html.parser')
                links = soup.select('a')
                for link in links:
                    if link.get('href') is not None:
                        if 'https://' in link.get('href'):
                            if link.get('href') not in secondary_urls:
                                secondary_urls.append(link.get('href'))
                        else:
                            if urljoin(url, link.get('href')) not in secondary_urls:
                                secondary_urls.append(urljoin(url, link.get('href')))
                return response, secondary_urls
            else:
                return response, []
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")


async def scrape_urls(urls: List[str], domains: List[str], full_domain: bool) -> str:
    """
    Scrapes URLs and extracts the combined text from the responses.

    Args:
        urls (List[str]): The list of URLs to scrape.
        domains (List[str]): The list of domain substrings to match against the URLs.
        full_domain (bool): Indicates whether to include the full domain in the extraction.

    Returns:
        str: The combined text from the scraped URLs.

    Raises:
        None

    """
    async with aiohttp.ClientSession() as session:
        results = []
        total_urls = list(set(urls))
        active_urls = list(set(urls))
        while active_urls:
            new_urls = []
            tasks = []
            for url in active_urls:
                tasks.append(crawl(session, url, domains, full_domain))
                await asyncio.sleep(0.1)
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for response in responses:
                if isinstance(response, tuple):
                    result, secondary = response
                    new_urls = new_urls + [url for url in secondary if url not in total_urls]
                    results.append(result)
            total_urls = total_urls + new_urls
            active_urls = new_urls
            cleaned_results = []
            for result in results:
                cleaned_results.append(clean(result))
            combined_text = "\n".join(cleaned_results)
        return combined_text


def reshape_and_list(text: str, separator: str) -> List[str]:
    """
    Reshapes the text and converts it into a list.

    Args:
        text (str): The text to reshape and convert.
        separator (str): The separator used to split the text.

    Returns:
        List[str]: The reshaped text as a list of strings.

    Raises:
        None

    """
    text = unidecode(text)
    text = re.sub(f'[^A-Za-z0-9$& \n-]+', ' ', text)
    text = re.sub(' +', ' ', text)
    text = list(set(text.split(separator)))
    return text


def len_filter(x: str) -> bool:
    """
    Filters the text based on its length.

    Args:
        x (str): The text to filter.

    Returns:
        bool: True if the text length is within the specified range, False otherwise.

    Raises:
        None

    """
    if (len(x) > 3) and (len(x) < 75):
        return True
    else:
        return False


def crawl_and_extract(url: str, full_domain: bool = False) -> pd.DataFrame:
    """
    Crawls a given URL, cleans the extracted text, and returns a DataFrame.

    Args:
        url (str): The URL to crawl and extract.
        full_domain (bool, optional): Indicates whether to include the full domain in the extraction. Default is False.

    Returns:
        pd.DataFrame: The DataFrame containing the extracted and cleaned text.

    Raises:
        None

    """
    out = pd.DataFrame()
    if '/products/' in url:
        domain = url.split('products')[0]
    else:
        print('Invalid URL')
        return
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(scrape_urls([url], [domain], full_domain))
    loop.close()
    out['cleaned_n'] = reshape_and_list(result, 'textbreakpoint')
    out['clean'] = out.cleaned_n.apply(lambda x: x.replace('\n', ''))
    out = out[out.clean.apply(lambda x: len_filter(x))]
    return out.drop('cleaned_n', axis=1)


def predict(df: pd.DataFrame, text_col: str, model: Model, threshold: float = 0.5) -> List[str]:
    """
    Predicts positive values based on a given DataFrame and model.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        text_col (str): The column name in the DataFrame containing the text data.
        model (Model): The Keras model for prediction.
        threshold (float, optional): The threshold for positive prediction. Default is 0.5.

    Returns:
        List[str]: The list of positive predictions.

    Raises:
        None

    """
    results_df = df.copy()
    results_df['prob'] = model.predict(df[[text_col]])
    positives = results_df[results_df.prob >= threshold][text_col].tolist()
    positives = list(set(positives))
    return positives


def ext_furniture(url: str, model: Model, full_domain: bool = False) -> list:
    """
    Extracts furniture information from a given URL.

    Args:
        url (str): The URL from which furniture information needs to be extracted.
        full_domain (bool, optional): Indicates whether to include the full domain in the extraction. Default is False.

    Returns:
        list: A list containing the extracted furniture information.

    """
    df = crawl_and_extract(url, full_domain)
    out = predict(df, 'clean', model)
    return out


def ans_bool(ans: str) -> bool:
    """
    Converts a string answer to a boolean value.

    Args:
        ans (str): The string answer.

    Returns:
        bool: The corresponding boolean value.


    """
    if ans == 'y':
        return True
    else:
        return False


def similarity_filtering(res_list: list, embedder: Model, thresh: float = 0.97) -> list:
    """
    Filters a list of results based on similarity using cosine similarity.

    Args:
        res_list (list): A list of results to be filtered.
        embedder (Model): The Keras model used for embedding.
        thresh (float, optional): The similarity threshold. Results with similarity scores above this threshold will be filtered out. Default is 0.97.

    Returns:
        list: A filtered list of results.

    Raises:
        None

    """
    emb_df = pd.DataFrame(embedder.predict(res_list), index=res_list).drop_duplicates()
    emb_df['sim_flag'] = 0
    help_df = emb_df.copy()
    for index, row in help_df.iterrows():
        row_ar = row.to_numpy()
        help_df = help_df.drop(index)
        for index_2, row_2 in help_df.iterrows():
            row_ar_2 = row_2.to_numpy()
            sim = dot(row_ar, row_ar_2) / (norm(row_ar) * norm(row_ar_2))
            if sim > thresh:
                emb_df.loc[index, 'sim_flag'] = 1
                break
    return emb_df[emb_df.sim_flag == 0].index.tolist()
