#libraries
from Xception_Model import Xception_Model
from PreProcessing import PreProcessing
from keras.models import model_from_json

if __name__ == '__main__':


    # Directory path for images
    Base_directory = '/kaggle/input/flame-dataset-fire-classification'
    test_path = 'Test/Test'
    Training_path = 'Training/Training'

    input_shape = (254, 254, 3)
    image_size = (254,254)
    batch = 16
    labels = ['Fire', 'No_Fire']

    # defining the full path for the files
    Full_Training_path = '{0}/{1}'.format(Base_directory, Training_path)
    Full_Test_path = '{0}/{1}'.format(Base_directory, test_path)

    #object of PreProcessing class
    pp =  PreProcessing()

    #image generators
    train_generator,validation_generator,test_generator = pp.image_generators(Full_Training_path=Full_Training_path,Full_Test_path=Full_Test_path,batch=batch,img_size=image_size)

    #object of Xception_Model Class
    Xception_mdl = Xception_Model()

    # If user wants to start training then press 1 else input 2
    mode = input("Please Enter 1 for Training, 2 for loading the saved Model for Evaluation")

    if int(mode)==1:

        # create the Xception Model
        model = Xception_mdl.create_Model(input_shape)
        #Train the model
        model,history = Xception_mdl.train_model(model,train_generator,validation_generator)

    elif int(mode)==2:

        # load json and create model
        json_file = open('{0}/{1}'.format(Base_directory, 'Xception_saved_model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # Load the saved weights
        model.load_weights('{0}/{1}'.format(Base_directory, 'Xception_saved_weights.h5'))

    else:
        print("wrong option please start again")

    if int(mode) in [1,2]:
        # evaluate the model
        Xception_mdl.model_evaluation(model, test_generator)


