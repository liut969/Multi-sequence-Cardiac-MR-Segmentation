from keras.callbacks import TensorBoard
#### globe



#### data.py
get_train_image_from_path = '../../data/c0t2lge'
get_train_image_save_path = '../../data/result/train_image'

get_train_label_from_path = '../../data/c0gt'
get_train_label_save_path = '../../data/result/train_label'

get_test_image_from_path = '../../data/c0t2lge'
get_test_image_save_path = '../../data/result/test_image'

predict_nii_gz_result_path = '../../data/result/predict_nii_gz_result'

train_start_num = 1
train_end_num = 36
test_start_num = 36
test_end_num = 46

################################################
#### main.py

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
# data_gen_args = dict()

batch_size = 16
train_path = '../../data/result'
classes_image = 'train_image'
classes_label = 'train_label'
flag_multi_class = True
num_class = 4
target_size = (256, 256)
# data_gen_save_to_dir = '../data/result/aug'
data_gen_save_to_dir = None

save_model_name = 'unet_model_100_9.hdf5'
steps_per_epoch = 100
epochs = 9

test_path = '../../data/result/test_image'
test_image_nums = 118
save_path = '../../data/result/test_label_for_predict'

tbCallBack = TensorBoard(log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_grads=True,
                 write_images=True,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)
################################################
#### model.py
input_size = (256, 256, 1)
