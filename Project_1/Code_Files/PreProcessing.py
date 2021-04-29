from tensorflow.keras.preprocessing.image import ImageDataGenerator

class PreProcessing:

    def __init__(self):
        self.validation_split=0.2  # train - validation split = 0.2


    def image_generators(self,Full_Training_path,Full_Test_path,batch,img_size):

        # loading the train images to imagegenerator. We keep 20% of the data for validation
        train_images = ImageDataGenerator(rotation_range=45,  # To rotate the image by max 45 degree
                                          horizontal_flip=True,  # Flip horizontally
                                          vertical_flip=True,  # flip vertically
                                          rescale=1.0 / 255,  # re-scale the RGB values between 0-1
                                          zoom_range=0.4,  # Zoom by factor 0.4
                                          shear_range=0.2,  # shear by factor 0.2
                                          fill_mode='nearest',  # fill the pixel by nearest value
                                          validation_split=self.validation_split)  # split the training set for validation - 20%

        train_generator = train_images.flow_from_directory(Full_Training_path,
                                                           target_size=img_size,  # Set tge target image size
                                                           color_mode='rgb',  # generate color images
                                                           class_mode='binary',  # target classes - 2
                                                           batch_size=batch,  # batch size set to 16
                                                           # shuffle=True,
                                                           subset='training')

        validation_generator = train_images.flow_from_directory(Full_Training_path,
                                                                target_size=img_size,
                                                                color_mode='rgb',
                                                                class_mode='binary',
                                                                batch_size=batch,
                                                                # shuffle=True,
                                                                subset='validation')

        test_images = ImageDataGenerator(rescale=1.0 / 255)  # re-scale the RGB values between 0-1

        test_generator = test_images.flow_from_directory(Full_Test_path,
                                                         target_size=img_size,
                                                         color_mode='rgb',
                                                         class_mode='binary',
                                                         shuffle=False,
                                                         batch_size=batch)

        return train_generator,validation_generator,test_generator
