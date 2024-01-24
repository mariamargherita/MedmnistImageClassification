
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class cnn_model:
    """
    This class creates our CNN model.
    """
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.build_model()

    def build_model(self, ):
        """
        Thia function builds the model.
        :return: model
        """

        input_data = layers.Input(shape=self.input_shape, name="input_layer")
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv2D_1')(input_data)
        x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2D_2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool')(x)
        x = layers.Dropout(0.25, name='dropout_1')(x)
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(128, activation='relu', name='output_dense')(x)
        x = layers.Dropout(0.5, name='dropout_2')(x)
        output_data = layers.Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=input_data, outputs=output_data, name='conv2D_model')
        print(self.model.summary())

    def fit_model(self, x_train, y_train, x_val, y_val, timestamp, batch_size, epochs):
        """
        This function fits the model.
        :param x_train: training set predictors
        :param y_train: training set labels
        :param x_val: validation set predictors
        :param y_val: validation set targets
        :param timestamp: timestamp
        :param batch_size: batch size
        :param epochs: number of epochs
        :return: model fit history
        """

        loss = tensorflow.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tensorflow.keras.optimizers.Adam() #RMSprop()

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        model_checkpoint = ModelCheckpoint(f'models/cp-best-{timestamp}.model',
                                           monitor='val_loss',
                                           mode='min',
                                           save_best_only=True,
                                           verbose=1)

        self.model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        history = self.model.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(x_val, y_val),
                                 callbacks=[model_checkpoint, early_stopping])
        return history

    def predict(self, x_test):
        """
        Returns model prediction on test set.
        :param x_test: testing set
        :return: labels prediction on test set
        """

        return self.model.predict(x_test)


class full_cnn_model:
    """
    This class creates our CNN model train on the whole training data.
    """

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.build_model()

    def build_model(self, ):
        """
        Thia function builds the model.
        :return: model
        """

        input_data = layers.Input(shape=self.input_shape, name="input_layer")
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv2D_1')(input_data)
        x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2D_2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool')(x)
        x = layers.Dropout(0.25, name='dropout_1')(x)
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(128, activation='relu', name='output_dense')(x)
        x = layers.Dropout(0.5, name='dropout_2')(x)
        output_data = layers.Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=input_data, outputs=output_data, name='conv2D_model')
        print(self.model.summary())

    def fit_model(self, x_train, y_train, batch_size, epochs):
        """
        This function fits the model.
        :param x_train: training set predictors
        :param y_train: training set labels
        :param batch_size: batch size
        :param epochs: number of epochs
        :return: model fit history
        """

        loss = tensorflow.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tensorflow.keras.optimizers.Adam()  # RMSprop()

        self.model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        history = self.model.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs)
        return history

    def predict(self, x_test):
        """
        Returns model prediction on test set.
        :param x_test: testing set
        :return: labels prediction on test set
        """

        return self.model.predict(x_test)

