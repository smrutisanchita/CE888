import tensorflow as tf
import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Flatten,Input

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score


class Xception_Model:

    def __init__(self):
        self.epoch = 1


    def create_Model(self,input_shape):

        # defining input tensor for the model
        img_input = Input(shape=input_shape)

        # load the pre-trainined Xception Model
        model = Xception(
            include_top=False,  # We will not include the top FC layers
            weights="imagenet",
            input_tensor=img_input,  # Input Tensor
            input_shape= input_shape,  # Image size
            pooling='avg')  # pooling method is avg

        # Fine Tuning of the Model

        last_layer = model.output  # Load the model Output
        x = Flatten(name='flatten')(last_layer)  # Flatten the layer
        x = Dense(1024, activation='relu', name='fc1')(x)
        out = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='linear')(x)  ## 2 classes
        model = Model(img_input, out)

        # we will train last 60 layers of the model and freeze rest
        for layer in model.layers[:-60]:
            layer.trainable = False

        #To print the summary of model
        print(model.summary())

        return model

    def train_model(self,model,train_generator,validation_generator):

        # have used Three early stopping criteria
        my_callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
            ModelCheckpoint(filepath='Xception_Model_{epoch:02d}.h5', save_best_only=False)]

        #compile the model
        model.compile(loss='hinge',
                      optimizer='AdaMax',
                      metrics='accuracy')

        history = model.fit(train_generator,
                            epochs=self.epoch,  # 50 epochs
                            validation_data=validation_generator,
                            callbacks=my_callbacks,
                            verbose=1)
        return model,history

    def model_evaluation(self,model, test_generator):
        # Make prediction on test set
        y_pred = model.predict(test_generator)
        # convert the output to binary form 0 and 1
        y_pred = [1 if x>0.5 else 0 for x in y_pred]

        # Actaul test class
        y_true = test_generator.classes

        print('Confusion Matrix:\n')
        print(confusion_matrix(y_true, y_pred))

        accuracy = accuracy_score(y_true, y_pred)
        print('Accuracy: %f' % accuracy)

        precision = precision_score(y_true, y_pred)
        print('Precision: %f' % precision)

        recall = recall_score(y_true, y_pred)
        print('Recall: %f' % recall)

        f1 = f1_score(y_true, y_pred)
        print('F1 score: %f' % f1)

