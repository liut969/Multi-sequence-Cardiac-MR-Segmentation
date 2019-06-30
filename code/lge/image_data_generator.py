"""
inference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
import numpy as np
import keras
from data_preprocess import *
import cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_images, train_labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.train_labels = train_labels
        self.train_images = train_images
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.train_images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.train_images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item


    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        for i, ID in enumerate(indexes):
            X[i, ] = self.train_images[ID]
            y[i, ] = self.train_labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


if __name__ == '__main__':
    data_preprocess = DataPreprocess(roi_x=256, roi_y=256, result_z=20)
    train_image = data_preprocess.get_roi_image(from_path_lge='../../data/c0t2lge', from_path_c0='../../data/result/c0gt_0_45', center_point_csv_path='./center_radii.csv', start_num=1, end_num=6)
    train_label = data_preprocess.get_roi_label(from_path_manual='../../data/lgegt', from_path_c0='../../data/result/c0gt_0_45', center_point_csv_path='./center_radii.csv', start_num=1, end_num=6)
    test_image = data_preprocess.get_roi_image(from_path_lge='../../data/c0t2lge', from_path_c0='../../data/result/c0gt_0_45', center_point_csv_path='./center_radii.csv', start_num=6, end_num=46)
    train_image = train_image[::, ::, ::, ::, np.newaxis]
    print(train_image.shape)
    print(train_label.shape)
    print(test_image.shape)

    params = {'dim': (20, 256, 256),
          'batch_size': 1,
          'n_classes': 4,
          'n_channels': 1,
          'shuffle': False}

    training_generator = DataGenerator(train_image, train_label, **params)

    for i, batch in enumerate(training_generator):
        print(i, len(batch))
        print(batch[0].shape, batch[1].shape)
        break



