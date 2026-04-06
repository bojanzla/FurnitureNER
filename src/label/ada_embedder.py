
# ADA EMBEDDER

import os
from dotenv import load_dotenv
import openai
import pandas as pd
from typing import List
import time
from src.utils.functions import trim


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = "text-embedding-ada-002"


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Gets the embedding vector for a given text using the specified model.

    Args:
        text (str): The text to get the embedding for.
        model (str, optional): The model name for embedding. Default is "text-embedding-ada-002".

    Returns:
        List[float]: The embedding vector.

    """
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']



if __name__ == "__main__":
    df_raw = pd.read_csv('../data/cleaned_frame.csv', index_col=0)
    df_raw.drop('cleaned_n', axis=1, inplace=True)
    df = trim(df_raw)
    df.to_csv('../data/prod_cleaned.csv')

    # This was done in step by step fashion due to frequent crashing of OpenAI APIs
    i = 0
    flag = True
    while flag:
        print(i)
        try:
            df_help = df[i: i + 100]
        except IndexError:
            df_help = df[i:]
        try:
            df_help['ada_embedding'] = df_help.clean.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
            df_help.to_csv(f'ada_embedded_{i}.csv', index=False)
            i = i + 100
            if i > df.shape[0]:
                flag = False
        except Exception as e:
            print(e)
            time.sleep(3)
            continue


