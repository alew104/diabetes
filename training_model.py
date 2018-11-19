from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from matplotlib import pyplot as plt

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

# dimensions of our images, 224x224 for VGG19
img_width, img_height = 224, 224

#directory data
model = 'VGG19'
#output h5
layers = 'VGG19'
#path to weights
weights_path = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

train_data_dir = model + '/training/'
validation_data_dir = model + '/validation/'
test_data_dir = model + '/testing/'

epochs = 50
batch_size = 16
input_shape = (img_width, img_height, 3)
#two classes: hemorrages or no hemorrhages
num_classes = 2


#begin vgg19 model
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())

# load weights of vgg19 no_top
model.load_weights(weights_path)

# add classifying layer
model.add(Dense(num_classes, activation='softmax'))

# freeze certain layers
model.layers[0].trainable = False
model.layers[1].trainable = False
model.layers[2].trainable = False
model.layers[3].trainable = False
model.layers[4].trainable = False
model.layers[5].trainable = False
model.layers[6].trainable = True
model.layers[7].trainable = True
model.layers[8].trainable = True
model.layers[9].trainable = True
model.layers[10].trainable = True
model.layers[11].trainable = True
model.layers[12].trainable = True
model.layers[13].trainable = True
model.layers[14].trainable = True
model.layers[15].trainable = True

# classifier layer
model.layers[16].trainable = True

# Compile Model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# show the model summary
model.summary()

# allow horizontal flipping and vertical flipping of training images
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True)

# do nothing with validation/test data
validation_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
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


model.save_weights('saved_weights/' + layers + '.h5')

# configure batch size and retrieve one batch of images
for X_batch, y_batch in train_generator:
	# create a grid of 3x3 images
	for i in range(0, 9):
		plt.subplot(330 + 1 + i)
		plt.imshow(X_batch[i].reshape(224, 224), cmap=plt.get_cmap('gray'))
	# show the plot
	plt.show()

plot_loss(history_obj.history['loss'], history_obj.history['val_loss'])
plot_acc(history_obj.history['acc'], history_obj.history['val_acc'])
