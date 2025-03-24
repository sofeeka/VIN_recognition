import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import kagglehub

ORIGINAL_CLASS_MAPPING = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
CHARS_TO_REMOVE = 'abdefghnqrtIOQ'
MODEL_SAVE_PATH = 'model.h5'


def log(msg):
    logging = True
    if logging:
        print(msg)


def get_raw_data():
    log('Downloading emnist train dataset...')
    path = kagglehub.dataset_download("crawford/emnist")

    log(f'Reading train data...')
    train_data_file_path = os.path.join(path, 'emnist-balanced-train.csv')
    train_df = pd.read_csv(train_data_file_path, header=None)

    log(f'Reading test data...')
    test_data_file_path = os.path.join(path, 'emnist-balanced-test.csv')
    test_df = pd.read_csv(test_data_file_path, header=None)

    return train_df, test_df


def get_class_mapping(original_class_mapping=ORIGINAL_CLASS_MAPPING, chars_to_remove=CHARS_TO_REMOVE):
    class_mapping = ''.join([c for c in original_class_mapping if c not in chars_to_remove])
    return class_mapping


def filter_data(df, original_class_mapping=ORIGINAL_CLASS_MAPPING, chars_to_remove=CHARS_TO_REMOVE):
    class_mapping = get_class_mapping(original_class_mapping, chars_to_remove)

    old_to_new_mapping = {original_class_mapping.index(c): class_mapping.index(c) if c in class_mapping else -1
                          for c in original_class_mapping}

    df.iloc[:, 0] = df.iloc[:, 0].map(old_to_new_mapping)
    df = df[df.iloc[:, 0] != -1].reset_index(drop=True)

    return df


def pad_and_transpose_data(df):
    padded_images = []

    for index, row in df.iterrows():
        label = row[0]
        flattened_image = row[1:].values

        image_28x28 = flattened_image.reshape(28, 28)
        image_30x30 = np.pad(image_28x28, pad_width=1, mode='constant', constant_values=255)
        image_30x30 = np.transpose(image_30x30, axes=[1, 0])
        flattened_padded_image = image_30x30.flatten()

        padded_image_with_label = np.insert(flattened_padded_image, 0, label)
        padded_images.append(padded_image_with_label)

    padded_df = pd.DataFrame(padded_images)
    return padded_df


def preprocess_data(df, num_classes, side):
    df_x = df.values[:, 1:]
    df_y = df.values[:, 0]

    df_x = df_x.reshape(-1, side, side, 1)
    df_x = df_x.astype('float32')
    df_x /= 255.0

    df_y = tf.keras.utils.to_categorical(df_y, num_classes=num_classes)

    return df_x, df_y


def get_emnist_data():
    log('Trying to get MNIST dataset...')
    raw_train_data, raw_test_data = get_raw_data()

    log('Filtering the data...')
    filtered_train_data = filter_data(raw_train_data)
    filtered_test_data = filter_data(raw_test_data)

    log('Padding and transposing the data...')
    padded_train_data = pad_and_transpose_data(filtered_train_data)
    padded_test_data = pad_and_transpose_data(filtered_test_data)

    num_classes = len(padded_train_data[0].unique())

    log('Preprocessing the data...')
    train_data_x, train_data_y = preprocess_data(padded_train_data, num_classes, 30)
    test_data_x, test_data_y = preprocess_data(padded_test_data, num_classes, 30)

    return train_data_x, train_data_y, test_data_x, test_data_y, num_classes


def define_model(num_classes):
    log('Building model...')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(30, 30, 1)))

    model.add(tf.keras.layers.Conv2D(18, (5, 5), strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (2, 2), padding='same', activation='relu'))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model


def visualize_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def load_or_train_model(save_path=MODEL_SAVE_PATH):
    if os.path.exists(save_path):
        log('Loading trained model...')
        model = tf.keras.models.load_model(save_path)

    else:
        train_x, train_y, test_x, test_y, num_classes = get_emnist_data()
        log(f'num classes {num_classes}')
        model = define_model(num_classes)

        log('Compiling model...')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        log('Training model...')
        history = model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y))

        log('Evaluating model...')
        loss, accuracy = model.evaluate(test_x, test_y)

        log(f"Test Loss: {loss}")
        log(f"Test Accuracy: {accuracy}")

        log('Saving model...')
        model.save(save_path)
        log(f"Model saved to {save_path}")

        visualize_history(history)

    return model


log('Waiting for the model...')
model = load_or_train_model()
