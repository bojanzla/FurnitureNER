
# LABEL, TRAIN AND SCRAPE SHARED FUNCTIONS DEPO

import pandas as pd
import re
import asyncio
import aiohttp
import bs4
from urllib.parse import urljoin
from unidecode import unidecode
from typing import List


def clean(html):
    """
    Cleans HTML by removing script and style elements and extracting text.

    Args:
        html (str): The HTML string to be cleaned.

    Returns:
        str: The cleaned text.

    """
    if html is None:
        return ''
    soup = bs4.BeautifulSoup(html, 'html.parser')

    # Kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    text = soup.get_text(separator='textbreakpoint')

    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


async def fetch(session, url):
    """
    Fetches the HTML content of a given URL using an async session.

    Args:
        session (aiohttp.ClientSession): The async session object.
        url (str): The URL to fetch.

    Returns:
        str: The HTML content of the URL.

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


def check_substring(strings, target_string):
    """
    Checks if any of the strings in a list is a substring of the target string.

    Args:
        strings (list): A list of strings to check.
        target_string (str): The target string to search.

    Returns:
        bool: True if a substring is found, False otherwise.

    """
    for string in strings:
        if target_string.startswith(string):
            return True
    return False


async def crawl(session, url, domains):
    """
    Crawls a URL to extract secondary URLs.

    Args:
        session (aiohttp.ClientSession): The async session object.
        url (str): The URL to crawl.
        domains (list): A list of allowed domains.

    Returns:
        tuple: A tuple containing the response and a list of secondary URLs.

    """
    secondary_urls = []
    if check_substring(domains, url):
        print(url)
        try:
            response = await fetch(session, url)

            soup = bs4.BeautifulSoup(response, 'html.parser')
            links = soup.select('a')

            for link in links:
                if link.get('href') is not None:
                    if 'https://' in link.get('href'):  # Absolute links
                        if link.get('href') not in secondary_urls:
                            secondary_urls.append(link.get('href'))
                    else:  # Relative links
                        if urljoin(url, link.get('href')) not in secondary_urls:
                            secondary_urls.append(urljoin(url, link.get('href')))

            return response, secondary_urls

        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")


async def scrape_urls(urls, domains):
    """
    Scrapes URLs and extracts text from their HTML content.

    Args:
        urls (list): A list of URLs to scrape.
        domains (list): A list of allowed domains.

    Returns:
        str: The combined text extracted from the URLs.

    """
    async with aiohttp.ClientSession() as session:
        results = []
        total_urls = list(set(urls))
        active_urls = list(set(urls))
        i = 0
        while active_urls:
            new_urls = []
            print(f'Loop {i}')
            tasks = []
            for url in active_urls:
                tasks.append(crawl(session, url, domains))
                await asyncio.sleep(0.1)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for response in responses:
                if isinstance(response, tuple):
                    result, secondary = response
                    new_urls = new_urls + [url for url in secondary if url not in total_urls]
                    results.append(result)

            total_urls = total_urls + new_urls
            active_urls = new_urls
            i += 1

            cleaned_results = []
            for result in results:
                cleaned_results.append(clean(result))

            combined_text = "\n".join(cleaned_results)

            with open("../../data/raw_text_train.txt", "w") as file:
                file.write(combined_text)

        return combined_text


def reshape_and_list(input_name, separator):
    """
    Reshapes the input text file and returns a list of unique strings.

    Args:
        input_name (str): The name of the input text file.
        separator (str): The separator used to split the text into strings.

    Returns:
        list: A list of unique strings.

    """
    with open(input_name) as f_in:
        text = f_in.read()
        text = unidecode(text)
        text = re.sub(f'[^A-Za-z0-9$& \n-]+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = list(set(text.split(separator)))
    return text


def trim(df):
    """
    Trims a DataFrame by cleaning and filtering the data.

    Args:
        df (pd.DataFrame): The DataFrame to be trimmed.

    Returns:
        pd.DataFrame: The trimmed DataFrame.

    """
    df_tag = df.copy()

    # Clean pure numbers
    df_tag['type'] = df_tag.clean.apply(lambda x: str(type(x)))
    df_tag = df_tag[df_tag.type == "<class 'str'>"]

    # Clean long and short strings
    df_tag['len'] = df_tag.clean.apply(lambda x: len(x))
    df_tag = df_tag.query('len < 70 and len > 3')
    df_tag = df_tag.drop(columns=['type', 'len'])
    return df_tag


def emb_set_builder(df, embedding_col, tag_col):
    """
    Builds an embedding set DataFrame from the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        embedding_col (str): The name of the column containing the embedding vectors.
        tag_col (str): The name of the column to be included in the embedding set DataFrame.

    Returns:
        pd.DataFrame: The embedding set DataFrame.

    """
    df_emb = pd.DataFrame()
    embedding_dim = len(df[embedding_col].iloc[0])
    for i in range(embedding_dim):
        df_emb[f'feature{i + 1}'] = df[embedding_col].apply(lambda x: x[i])
    df_emb[tag_col] = df[tag_col]
    return df_emb


def f1_score_list(precision_list: List[float], recall_list: List[float]) -> List[float]:
    """
    Calculates the F1 score given precision and recall lists.

    Args:
        precision_list: List of precision values.
        recall_list: List of recall values.

    Returns:
        List of F1 scores.
    """
    f1_score_list = []
    i = 0
    for i in range(len(precision_list)):
        f1_score_list.append(2 * (precision_list[i] * recall_list[i]) / (precision_list[i] + recall_list[i]))
        i = i + 1
    return f1_score_list
