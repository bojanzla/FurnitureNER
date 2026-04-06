
# MAIN

import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


from app_utils.utils import ans_bool, ext_furniture, similarity_filtering

# Load prediction model
model = tf.keras.models.load_model("../../models/bert_L-2_H-768_A-12_hidden.keras", custom_objects={'KerasLayer': hub.KerasLayer})

# embedder
embedder = tf.keras.models.load_model("../../models/bert_embedder.keras", custom_objects={'KerasLayer': hub.KerasLayer})

# Main function
if __name__ == "__main__":
    os.system('clear')
    one_more = True
    while one_more:
        url = input('Please enter URL for furniture product extraction: ')
        if 'product' not in url:
            print('URL must contain "product"')
            continue
        full_domain = input('Do you want to scrape full domain? (y/n): ')
        full_domain = ans_bool(full_domain)
        raw_res = ext_furniture(url, model, full_domain=full_domain)
        if not raw_res:
            print('No products found')
            one_more = input('One more? (y/n): ')
            one_more = ans_bool(one_more)
            continue
        # Similarity filtering with untrained embeddings
        raw_res = similarity_filtering(raw_res, embedder)
        print(raw_res)
        one_more = input('One more? (y/n): ')
        one_more = ans_bool(one_more)
