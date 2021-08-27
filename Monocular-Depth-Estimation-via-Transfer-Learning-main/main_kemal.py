import numpy as np
import tensorflow as tf
import argparse
from PIL import Image # Imports PIL module
from io import BytesIO
from tensorflow.keras.utils import Sequence
from skimage.transform import resize
from tensorflow.keras.applications import DenseNet169
import sklearn
import os
import matplotlib.pyplot as plt

from skimage import io
from zipfile import ZipFile

###### (Kemal) #####
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Concatenate, UpSampling2D
import tensorflow.keras.backend


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# python train.py --data nyu
parser = argparse.ArgumentParser(description='My first complete deep learning code') #Input parameters
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')#The batch size of the training network
parser.add_argument('--max_depth', type=int, default=1000, help='The maximal depth value')#The max depth of the images
parser.add_argument('--data', default="nyu", type=str, help='Training dataset.')#A default train dataset
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')#GPU number
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')


args = parser.parse_args() #Add input as parameters



def _parse_function(filename, label):
    # Read images from disk
    shape_rgb = (512, 512, 3)
    shape_depth = (512, 512, 1)
    image_decoded = tf.image.decode_jpeg(tf.io.read_file(filename))
    depth_resized = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(label)),
                                    [shape_depth[0], shape_depth[1]])

    # Format
    rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    depth = tf.image.convert_image_dtype(depth_resized / 255.0, dtype=tf.float32)

    # Normalize the depth values (in cm)
    depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)

    return rgb, depth

if args.data == 'nyu':
    ## Train_dataset
    csv_file = 'D:/MATLAB/Deep_learnning_based_conformal_NRSfM/Data_generation/data/data_train.csv'
    csv = open(csv_file, 'r').read()
    nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))
    nyu2_train = sklearn.utils.shuffle(nyu2_train, random_state=0)
    filenames = [os.path.join('D:/MATLAB/Deep_learnning_based_conformal_NRSfM/Data_generation',i[0]) for i in nyu2_train]
    labels = [os.path.join('D:/MATLAB/Deep_learnning_based_conformal_NRSfM/Data_generation',i[1])for i in nyu2_train]
    length = len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=len(filenames), reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.map(map_func=_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    batch_size = args.batch_size  # batch_size from inputs, default value is 2
    train_generator = dataset.batch(batch_size=batch_size)
    ## Test_dataset
    csv_file_test = 'D:/MATLAB/Deep_learnning_based_conformal_NRSfM/Data_generation/data/data_test.csv'
    csv_test = open(csv_file_test, 'r').read()
    nyu2_test = list((row.split(',') for row in (csv_test).split('\n') if len(row) > 0))
    nyu2_test = sklearn.utils.shuffle(nyu2_test, random_state=0)
    filenames_test = [os.path.join('D:/MATLAB/Deep_learnning_based_conformal_NRSfM/Data_generation', i[0]) for i in nyu2_test]
    labels_test = [os.path.join('D:/MATLAB/Deep_learnning_based_conformal_NRSfM/Data_generation', i[1]) for i in nyu2_test]
    length_test = len(filenames_test)
    dataset_test = tf.data.Dataset.from_tensor_slices((filenames_test, labels_test))
    dataset_test = dataset_test.shuffle(buffer_size=len(filenames_test), reshuffle_each_iteration=True)
    dataset_test = dataset_test.repeat()
    dataset_test = dataset_test.map(map_func=_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    batch_size = args.batch_size  # batch_size from inputs, default value is 2
    test_generator = dataset_test.batch(batch_size=batch_size)

######################### Network encoder-decoder (Kemal) ##########################
def model(existing='', is_twohundred=False, is_halffeatures=True):
        
    print('Loading base model (DenseNet)..')

    # Encoder Layers
    base_model = DenseNet169(input_shape=(None, None, 3), include_top=False, weights='imagenet')

    print('Base model loaded.')

    # Starting point for decoder
    base_model_output_shape = base_model.layers[-1].output.shape

    # Layer freezing?
    for layer in base_model.layers: layer.trainable = True

    # Starting number of decoder filters
    decode_filters = int(int(base_model_output_shape[-1])/2)

    # Define upsampling layer
    def upproject(tensor, filters, name, concat_with):
        up_i = UpSampling2D(size=(2, 2), interpolation='bilinear', name=name+'_upsampling2d')(tensor)
        up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        return up_i

    # Decoder Layers
    decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)

    decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
    decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
    decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
    decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')

    # Extract depths (final layer)
    conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

    # Create the model
    m = Model(inputs=base_model.input, outputs=conv3)

    print('Model created.')
    
    return m

print(model.summary())


######################### Multi-gpu setup:################################
basemodel = model
if args.gpus > 1: model = tf.keras.utils.multi_gpu_model(model, gpus=args.gpus)

######################### Loss Function (Kemal) ################################

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    
    # Point-wise depth (point-wise L1 loss)
    l_depth = tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true), axis=-1)

    # Edges (L1 loss defined over the image gradient)
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = tf.keras.backend.mean(tf.keras.backend.abs(dy_pred - dy_true) + tf.keras.backend.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = tf.keras.backend.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = theta
    w2 = 1.0
    w3 = 1.0

    return (w1 * tf.keras.backend.mean(l_depth)) + (w2 * tf.keras.backend.mean(l_edges)) + (w3 * l_ssim)



print('\n\n\n', 'Compiling model..')
######################### Trainning ################################
learning_rate=0.0001
model.compile(optimizer=tf.optimizers.Adam(1e-2,lr=learning_rate, amsgrad=True),loss=depth_loss_function)
print('\n\n\n', 'Compiling complete')


model.fit(train_generator,epochs=args.epochs,steps_per_epoch=length//batch_size)
###########################Save model###############################
model.save("./models/model_with.h5", include_optimizer=False)

model.save('./models/', save_format='tf',include_optimizer=False)

##########################Result test################################
score=model.evaluate(test_generator,steps=10)

print("last score:",score)

#########################Predict a result#############################
image_decoded = tf.image.decode_jpeg(tf.io.read_file('1.jpg'))
rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
rgb=np.expand_dims(rgb, axis = 0)
#model = tf.keras.models.load_model('./models/model.h5',custom_objects={'depth_loss_function': depth_loss_function})
result=model.predict(rgb)
#print(result)
image_new=result[0,:,:,0]
plt.imshow(image_new)
plt.show()