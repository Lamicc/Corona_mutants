import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_data():
    print("\nLoading the dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape)
    return x_train, x_test, y_train, y_test


def load_data_mut(filename_train='data/temporal_train_mat.npy', filename_test='data/temporal_test_mat.npy'):
    print("\nLoading the dataset...")
    x_train = np.load(filename_train)
    x_test = np.load(filename_test)
    print(x_train.shape)
    print(x_test.shape)
    return x_train, x_test


def preprocess_data(x_train, x_test, normalisation=1, validation_prop=10):
    print("\nPreprocessing the dataset...")
    num_samples_train, num_rows_train, num_cols_train = x_train.shape
    num_samples_test, num_rows_test, num_cols_test = x_test.shape

    x_train = x_train.reshape(num_samples_train, num_rows_train * num_cols_train).astype('float32') / normalisation
    x_test = x_test.reshape(num_samples_test, num_rows_test * num_cols_test).astype('float32') / normalisation

    num_valid = int(np.floor(num_samples_train / validation_prop))
    print("\tNumber of validation samples: ", num_valid)

    x_val = x_train[-num_valid:]
    x_train = x_train[:-num_valid]

    print("\tTraining shape: \t", x_train.shape)
    print("\tTest shape: \t\t", x_test.shape)
    print("\tValidation shape: \t\t\t", x_val.shape)
    print("\tFeature vector size: \t\t", num_rows_train * num_cols_train)

    return x_train, x_test, x_val


def setup_network(input_dim, hidden_dim, code_dim, loss='binary_crossentropy', optimizer='adam'):
    print("\nSetting up the network...")

    # Create Network
    inputs = tf.keras.Input(shape=(input_dim,), name='sequences')

    hidden_1 = tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    code = tf.keras.layers.Dense(code_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(hidden_1)
    hidden_2 = tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(code)
    outputs = tf.keras.layers.Dense(input_dim, activation='sigmoid', name='predictions', kernel_regularizer=tf.keras.regularizers.l2(0.001))(hidden_2)

    #hidden_1 = tf.keras.layers.Dense(hidden_dim, activation='relu')(inputs)
    #code = tf.keras.layers.Dense(code_dim, activation='relu')(hidden_1)
    #hidden_2 = tf.keras.layers.Dense(hidden_dim, activation='relu')(code)
    #outputs = tf.keras.layers.Dense(input_dim, activation='sigmoid', name='predictions')(hidden_2)

    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.summary()

    return autoencoder


def run_network(autoencoder, train_data, train_data_labels, validation_data, validation_data_labels,
                filename='plots/check', epochs=5, batchsize=64):
    print("\nTraining the network...")
    history = autoencoder.fit(train_data, train_data_labels, epochs=epochs, batch_size=batchsize, shuffle=True,
                              validation_data=(validation_data, validation_data_labels))
    return history


def plot_training_loss(history, filename='plots/training_loss_10k.png'):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Training Loss', color='b')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='r')
    plt.title('Training and Validation Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc="upper right")
    plt.savefig(filename)


def evalute_on_test(autoencoder, test_data, loss='binary_crossentropy', filename_predictions='plots/predictions.npy',
                    filename_test_loss='plots/test_loss_per_sample.npy', filename_histogram='plots/test_loss.png'):
    print("\nMaking predictions...")
    predictions = autoencoder.predict(test_data)
    np.save(filename_predictions, predictions)

    print('\nEvaluating on test data...')
    results = autoencoder.evaluate(test_data, test_data, batch_size=128)
    print('test loss, test acc:', results)

    test_loss_per_sample = []
    for i in range(len(test_data)):
        test_loss_per_sample.append(
            autoencoder.evaluate(test_data[i, None], test_data[i, None], batch_size=1, verbose=0))
    test_loss_per_sample = np.array(test_loss_per_sample)
    test_loss_total = np.mean(test_loss_per_sample)

    print('\nMean of test losses: ', test_loss_total)
    print("Minimum loss: \t", np.min(test_loss_per_sample))
    print("Maximum loss: \t", np.max(test_loss_per_sample))
    np.save(filename_test_loss, test_loss_per_sample)

    plt.figure()
    plt.hist(test_loss_per_sample, bins=100)
    plt.xlim([min(test_loss_per_sample), max(test_loss_per_sample)])
    plt.title("Histogram of Test Loss. Mean: %.5f, Min: %.5f. Max: %.5f" % (
    np.mean(test_loss_per_sample), np.min(test_loss_per_sample), np.max(test_loss_per_sample)))
    plt.xlabel("Loss")
    plt.savefig(filename_histogram)

    return test_loss_total, test_loss_per_sample


# train_data, test_data, _, _ = load_data()
# train_data, test_data, validation_data = preprocess_data(train_data, test_data, normalisation=255)


train_data, test_data = load_data_mut(filename_train='data/temporal_train_mat.npy_20000_12000.npy',
                                      filename_test='data/temporal_test_mat.npy_20000_12000.npy')
train_data, test_data, validation_data = preprocess_data(train_data, test_data, normalisation=1)

loss = 'binary_crossentropy' # 'mse'
autoencoder = setup_network(input_dim=train_data.shape[1], hidden_dim=128, code_dim=32, loss=loss)
history = run_network(autoencoder, train_data=train_data, train_data_labels=train_data, validation_data=validation_data,
                      validation_data_labels=validation_data, epochs=50, batchsize=64)

plot_training_loss(history, filename='plots/training_loss_20000_12000_reg_new.png')
np.save('plots/training_loss_reg_new.npy', history.history['loss'])
np.save('plots/validation_loss_reg_new.npy', history.history['val_loss'])

test_loss_total, test_loss_per_sample = evalute_on_test(autoencoder, test_data, loss=loss,
                                                        filename_predictions='plots/predictions_20000_12000_reg_new.npy',
                                                        filename_test_loss='plots/test_loss_per_sample_20000_12000_reg_new.npy',
                                                        filename_histogram='plots/test_loss_20000_12000_reg_new.png')
