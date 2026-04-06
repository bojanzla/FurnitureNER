
# BERT EMBEDDER

import ast
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from src.config import map_name_to_handle, map_model_to_preprocess


def build_embedder_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    return tf.keras.Model(text_input, net)

tf.get_logger().setLevel('ERROR')

bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)

if __name__ == "__main__":
    df = pd.read_csv('../../data/val_set_ada.csv', index_col=0)
    df = df.rename(columns={"ada_embedding": "ada"})
    df['ada'] = df['ada'].apply(ast.literal_eval)

    column_data = df['clean'].values
    column_data = column_data.reshape(-1, 1)
    print(column_data.shape)

    embedder_model = build_embedder_model()
    # embedder_model.save(f'../../models/bert_embedder.keras')
    bert_raw_result = embedder_model.predict(column_data)

    df['bert'] = bert_raw_result.tolist()

    df.to_csv('../../data/val_set_emb.csv')
    df.to_pickle('../../data/val_set_emb.pkl')
    print(df.head())
