
# BERT FINE TUNING

import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from src.config import map_name_to_handle, map_model_to_preprocess
from src.utils.functions import f1_score_list


def fine_tuner(df_train: pd.DataFrame, df_test: pd.DataFrame, text_col: str, tag_col: str,
               epochs: int = 10, hidden: bool = False) -> None:
    """
    Fine-tunes a BERT model for binary classification.

    Args:
        df_train: Training data DataFrame.
        df_test: Test data DataFrame.
        text_col: Name of the column containing the text.
        tag_col: Name of the column containing the target labels.
        epochs: Number of training epochs (default: 10).
        hidden: Whether to include a hidden layer in the model (default: False).

    Returns:
        None
    """

    # Load the data
    train = df_train[[text_col, tag_col]].dropna()
    test = df_test[[text_col, tag_col]].dropna()

    # Test-train split
    X_train = train.drop('tag', axis=1)
    y_train = train[['tag']]
    X_test = test.drop('tag', axis=1)
    y_test = test[['tag']]

    # Creating suffix for naming the model with hidden layers
    if hidden:
        suffix = '_hidden'
    else:
        suffix = ''

    # Choosing the BERT model
    bert_model_name = 'small_bert/bert_en_uncased_L-8_H-128_A-2'
    save_name = f'bert_L-8_H-128_A-2{suffix}'

    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
    # tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3'
    # tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3'

    # Early stop
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=9, min_delta=0.01,
                                                mode='max', restore_best_weights=True)

    # Layers
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    if hidden:   # hidden layers
        net = tf.keras.layers.Dropout(0.2)(net)
        net = tf.keras.layers.Dense(64, activation='relu', name="hiden",
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)

    # Use inputs and outputs to construct a final model
    model = tf.keras.Model(inputs=[text_input], outputs=[net])

    # Model summary
    print(model.summary())
    tf.keras.utils.plot_model(model)

    # Chossing metrics to track
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics)

    history = model.fit(x=X_train, y=y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs,
                        callbacks=[callback])

    # Train
    model.save(f'../../models/{save_name}.keras')

    # Training results
    history_dict = history.history
    loss, accuracy, precision, recall, auc = model.evaluate(X_test, y_test)

    print('\n')
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {2 * (precision * recall) / (precision + recall)}')


    precision = history_dict['precision']
    recall = history_dict['recall']
    f1_score = f1_score_list(precision, recall)

    val_precision = history_dict['val_precision']
    val_recall = history_dict['val_recall']
    val_f1_score = f1_score_list(val_precision, val_recall)

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    # Plot metrics
    plt.rcParams['figure.constrained_layout.use'] = True
    epochs = range(1, len(precision) + 1)
    fig = plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, f1_score, 'r', label='Training f1 score')
    plt.plot(epochs, val_f1_score, 'b', label='Validation f1 score')
    plt.title('Training and validation f1 score')
    plt.xlabel('Epochs')
    plt.ylabel('f1 score')
    plt.legend(loc='lower right')

    plt.savefig(f'../../reports/{save_name}.png')
    plt.show()


def evaluate_bert(test: pd.DataFrame, model_name: str) -> None:
    """
    Evaluate a BERT model on the test dataset.

    Args:
        test: DataFrame containing the test data.
        model_name: Name of the BERT model.

    Returns:
        None
    """

    model = tf.keras.models.load_model(f"../../models/{model_name}.keras",
                                       custom_objects={'KerasLayer': hub.KerasLayer})
    print(model.metrics_names)
    y_test = test[['tag']]
    X_test = test[['clean']]
    y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=['probability_1'])
    eval_df = y_test.join(y_pred)

    # Threshold optimization
    prob_list = [0.3, 0.4, 0.5, 0.6, 0.7]

    for prob in prob_list:
        df_res_extreme = eval_df.copy()
        df_res_extreme['cut_label'] = df_res_extreme['probability_1'].apply(lambda x: 1 if x >= prob else 0)
        y_test_ext = df_res_extreme.tag
        y_pred_ext = df_res_extreme.cut_label

        print(f'Probability: {prob}')
        print("\n=== Confusion Matrix ===")
        print(confusion_matrix(y_test_ext, y_pred_ext))
        print("\n=== Classification Report ===")
        print(classification_report(y_test_ext, y_pred_ext, zero_division=0))
        print(50 * '-' + 'n')


if __name__ == "__main__":
    train = pd.read_pickle('../../data/tagged_set_emb.pkl')
    test = pd.read_pickle('../../data/val_set_emb.pkl')
    # fine_tuner(train, test, 'clean', 'tag', epochs=24, hidden=True)
    # evaluate_bert(test, 'bert_L-8_H-128_A-2')




