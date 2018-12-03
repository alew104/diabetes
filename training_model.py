from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import optimizers, models, Model
from keras.applications import VGG19
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

# directory data
model = 'VGG19'
# output h5
layers = 'VGG19'
# path to weights
weights_path = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

train_data_dir = model + '/training/'
validation_data_dir = model + '/validation/'
test_data_dir = model + '/testing/'

epochs = 100
batch_size = 32
input_shape = (img_width, img_height, 3)
# two classes: hemorrages or no hemorrhages
num_classes = 2


# keras vgg 19
# model = Sequential()
# model.add(VGG19(include_top=False, pooling='avg', weights='imagenet', input_shape=input_shape))
# model.layers[0].trainable = False


vgg19 = VGG19(VGG19(include_top=False, pooling='avg', weights='imagenet', input_shape=input_shape))

vgg19.get_layer('block1_conv1').trainable = False
vgg19.get_layer('block1_conv2').trainable = False
vgg19.get_layer('block1_pool').trainable = False

vgg19.get_layer('block2_conv1').trainable = False
vgg19.get_layer('block2_conv2').trainable = False
vgg19.get_layer('block2_pool').trainable = False

vgg19.get_layer('block3_conv1').trainable = False
vgg19.get_layer('block3_conv2').trainable = False
vgg19.get_layer('block3_conv3').trainable = False
vgg19.get_layer('block3_conv4').trainable = True
vgg19.get_layer('block3_pool').trainable = True

vgg19.get_layer('block4_conv1').trainable = True
vgg19.get_layer('block4_conv2').trainable = True
vgg19.get_layer('block4_conv3').trainable = True
vgg19.get_layer('block4_conv4').trainable = True
vgg19.get_layer('block4_pool').trainable = True

vgg19.get_layer('block5_conv1').trainable = True
vgg19.get_layer('block5_conv2').trainable = True
vgg19.get_layer('block5_conv3').trainable = True
vgg19.get_layer('block5_conv4').trainable = True
vgg19.get_layer('block5_pool').trainable = True

x = vgg19.get_layer('block5_pool').output
x = Flatten()(x)
x = Dense(10, activation='softmax')(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(vgg19.input, x)

# Compile Model
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

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
            steps_per_epoch=batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=2)

model.save('saved_models/' + layers + '.h5')

plot_loss(history_obj.history['loss'], history_obj.history['val_loss'])
plot_acc(history_obj.history['acc'], history_obj.history['val_acc'])
