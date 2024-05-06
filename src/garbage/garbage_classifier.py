import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large


IMG_SIZE = 224


class GarbageClassifier:

    class_names = None
    batch_size = None
    val_split = 0.15

    def __init__(self, class_names, batch_size, tracker=None):
        self.class_names = class_names
        self.batch_size = batch_size
        self.tracker = tracker

    def load_dataset(self, train_path, validation_path):
        train_gen = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
                shear_range=0.2,
                zoom_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                validation_split=self.val_split)
        self.train_batches = train_gen.flow_from_directory(directory=train_path, target_size=(224, 224), classes=self.class_names,
                                                           shuffle=True, seed=42, batch_size=self.batch_size, subset='training')

        # extract images to validation set
        valid_gen = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
                validation_split=self.val_split)
        self.valid_batches = valid_gen.flow_from_directory(directory=validation_path,
                                                           target_size=(224, 224),
                                                           classes=self.class_names,
                                                           shuffle=False,
                                                           seed=42,
                                                           batch_size=self.batch_size,
                                                           subset='validation')

        self.tracker.track_config(
            {
                "val_split": self.val_split,
                "batch_size": self.batch_size,
                "dataset_size": self.valid_batches.samples + self.train_batches.samples
            }
        )

        # Print the number of images per class for training and validation sets
        print("\nTraining set distribution:")
        for i, class_name in enumerate(self.train_batches.class_indices):
            print(f"{class_name}: {np.sum(self.train_batches.classes == i)} images")
        
        print("\nValidation set distribution:")
        for i, class_name in enumerate(self.valid_batches.class_indices):
            print(f"{class_name}: {np.sum(self.valid_batches.classes == i)} images")
        
    def build_model(self, hidden1_size=512, hidden2_size=128, l2_param=0.002, dropout_factor=0.2, bias_regularizer='l1'):
        # set the input image size for proposed CNN model
        self.IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

        # import the convolution base of the MobileNetV3 model with pre-trained weights
        self.base_model = tf.keras.applications.MobileNetV3Large(
                                                input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet'
                                            )
        self.model = Sequential()

        # Freeze the convolutional base of MobileNet to prevent the pre-trained weights being updated
        # during training in order to extract features
        self.base_model.trainable = False

        # add MobileNet convolution base to initialize sequential model
        self.model.add(self.base_model)

        # add global average pooling layer
        self.model.add(GlobalAveragePooling2D())

        l2_reg = tf.keras.regularizers.l2(l2=l2_param)

        # add densely-connected NN layer with 512 hidden units
        self.model.add(Dense(units=hidden1_size, activation='relu', kernel_regularizer=l2_reg,
                             bias_regularizer=bias_regularizer))  # use ReLU activation function
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_factor))

        # add densely-connected NN layer with 128 hidden units
        self.model.add(Dense(units=hidden2_size, activation='relu', kernel_regularizer=l2_reg,
                             bias_regularizer=bias_regularizer))  # use ReLU activation function
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_factor))

        # add densely-connected NN layer with 6 hidden units
        self.model.add(Dense(units=6, activation='softmax'))  # use Softmax activation function to do final predictions

        self.tracker.track_config(
            {
                "base_model": "MobileNetV3Large",
                "image_size": IMG_SIZE,
                "l2_param": l2_param,
                "hidden_layer1_size": hidden1_size,
                "hidden_layer2_size": hidden2_size,
                "use_bias_regularizer": bias_regularizer is not None,

            }
        )

    def compile(self, learning_rate=0.0005, loss='categorical_crossentropy', min_lr=0.00001, decrease_factor=0.2, patience=5,):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])

        self._callbacks = []
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=decrease_factor,
                                                         patience=patience, min_lr=min_lr)
        self._callbacks.append(reduce_lr)

        tracker_callback = self.tracker.get_callback()
        if tracker_callback:
            self._callbacks.append(tracker_callback)

        self.tracker.track_config(
            {
                "loss": "Categorical Cross Entropy" if loss == 'categorical_crossentropy' else "Focal Loss",
                "lr": learning_rate,
                "optimizer": "Adam",
                "reduceOnPlateau": True,
                "reduceOnPlateauMinLR": min_lr,
                "reduceOnPlateauDecreaseFactor": decrease_factor,
                "reduceOnPlateauPatience": patience
            }
        )

    def fit(self, epochs=30):
        return self.model.fit(x=self.train_batches, validation_data=self.valid_batches, epochs=epochs,
                              callbacks=self._callbacks, verbose=2)

    def predict(self, batch_generator):
        batch_runs = batch_generator.samples // batch_generator.batch_size + 1
        return self.model.predict(batch_generator, batch_runs)

    def save(self, save_model_path):
        return self.model.save(save_model_path)
        
    def generate_embeddings(self, training_batches):
        """Generates embeddings using the pretrained MobileNetV3 model."""
        if not hasattr(self, 'embedding_model'):
            # Define el modelo de embedding si aún no se ha definido
            base_model = MobileNetV3Large(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, pooling='avg', weights='imagenet')
            self.embedding_model = Model(inputs=base_model.input, outputs=base_model.output)
        
        # Generar embeddings para los batches de entrenamiento
        return self.embedding_model.predict(training_batches, steps=len(training_batches))
    
    def save_training_embeddings(self,
                                 training_batches,
                                 train_embeddings_path=None,
                                 train_target=None,
                                 train_prediction=None,
                                 index_to_class=None,
                                 include_labels=False):
        """Saves the training embeddings along with optional labels to a specified path."""
        # Generar embeddings
        embeddings = self.generate_embeddings(training_batches)
        df_embeddings = pd.DataFrame(embeddings)

        if include_labels:
            class_indices = self.train_batches.class_indices
            index_to_class = {v: k for k, v in class_indices.items()}
            
            df_embeddings['target'] = train_target
            df_embeddings['prediction'] = np.argmax(train_prediction, axis=1)
            df_embeddings['prediction_score'] = np.max(train_prediction, axis=1)
            df_embeddings['class_label'] = df_embeddings['target'].map(self.index_to_class)
        
        df_embeddings.columns = df_embeddings.columns.astype(str)

        if train_embeddings_path:
            df_embeddings.to_parquet(train_embeddings_path)

        return df_embeddings

