from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications import VGG16, ResNet50, InceptionV3, VGG19, InceptionResNetV2
from keras import optimizers
import numpy as np
from matplotlib import pyplot as plt


# dimensions of our images, 224x224 for VGG16
img_width, img_height = 224, 224

#directory data
model = 'VGG16'
#output h5
layers = 'ResNet50'

train_data_dir = '/Users/alex/Desktop/' + model + '/training/'
validation_data_dir = '/Users/alex/Desktop/' + model + '/validation/'
test_data_dir = '/Users/alex/Desktop/' + model + '/testing/'

epochs = 50
batch_size = 16
#two classes: hemorrages or no hemorrhages
num_classes = 2
input_shape = (img_width, img_height, 3)

model = Sequential()
#model.add(VGG16(include_top=False, pooling='avg', weights='imagenet', input_shape=input_shape), classes=num_classes)
#model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape=input_shape))
model.add(VGG19(include_top=False, pooling='avg', weights='imagenet', input_shape=input_shape))
#model.add(InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_shape=input_shape, classes=num_classes))
#model.add(InceptionResNetV2(include_top=False, pooling='avg', weights='imagenet', input_shape=input_shape, classes=num_classes))
model.add(Dense(num_classes, activation='softmax'))

#freeze VGG19 layer since it is already trained
model.layers[0].trainable = False

# Compile Model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#show the model summary
model.summary()

# allow horizontal flipping of images
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True)

# do nothing with test data
test_datagen = ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

history_obj = model.fit_generator(
            train_generator,
            steps_per_epoch=10,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=2)


# configure batch size and retrieve one batch of images
for X_batch, y_batch in train_generator:
	# create a grid of 3x3 images
	for i in range(0, 9):
		plt.subplot(330 + 1 + i)
		plt.imshow(X_batch[i].reshape(224, 224), cmap=pyplot.get_cmap('gray'))
	# show the plot
	plt.show()


# To plot both training and testing cost/loss ratio
def plot_loss(train_loss, valid_loss):
    fig, ax = plt.subplots()
    fig_size = [12, 9]
    plt.rcParams["figure.figsize"] = fig_size
    plt.plot(train_loss, label='Training Cost')
    plt.plot(valid_loss, label='Validation Cost')
    plt.title('Cost over time during training')
    legend = ax.legend(loc='upper right')
    plt.show()


# To plot both training and validation accuracy
def plot_acc(train_acc, valid_acc):
    fig, ax = plt.subplots()
    fig_size = [12, 9]
    plt.rcParams["figure.figsize"] = fig_size
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(valid_acc, label='Validation Accuracy')
    plt.title('Accuracy over time during training')
    legend = ax.legend(loc='upper right')
    plt.show()

plot_loss(history_obj.history['loss'], history_obj.history['val_loss'])
plot_acc(history_obj.history['acc'], history_obj.history['val_acc'])

model.save_weights('/Users/alex/Desktop/' + layers + '.h5')

model.load_weights('/Users/alex/Desktop/' + layers + '.h5')
#arr = model.predict_generator(
#        test_generator,
#        verbose=1
#)
