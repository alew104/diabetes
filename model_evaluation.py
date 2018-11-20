# Modify 'test1.jpg' and 'test2.jpg' to the images you want to predict on

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys

def run(image):
    # dimensions of our images
    img_width, img_height = 224, 224

    #directory data
    model = 'VGG19'

    # model weights
    weights_path = "saved_weights/VGG19.h5"

    # testing data directory
    test_data_dir = model + '/testing/'

    # load the model we saved
    model = load_model(weights_path)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # predicting images
    img = image.load_img(test_data_dir + image, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    print classes

    # predicting multiple images at once
    #img = image.load_img('test2.jpg', target_size=(img_width, img_height))
    #y = image.img_to_array(img)
    #y = np.expand_dims(y, axis=0)

    # pass the list of multiple images np.vstack()
    #images = np.vstack([x, y])
    #classes = model.predict_classes(images, batch_size=10)

    # print the classes, the images belong to
    print classes
    print classes[0]
    print classes[0][0]

def main(image):
    run(image)


if __name__ == '__main__':
    main(sys.argv[1])
