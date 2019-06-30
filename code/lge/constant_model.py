from keras.callbacks import TensorBoard
#### globe



#### data_preprocess.py
get_train_image_from_path = '../../data/c0t2lge'
save_path_train_image = '../../data/result_1/train_image_roi'
#
get_train_label_from_path = '../../data/lgegt'
save_path_train_label = '../../data/result_1/train_label_roi'

get_test_image_from_path = get_train_image_from_path
save_path_test_image = '../../data_result_1/test_image_roi'

from_path_c0 = '../../data/result/c0gt_0_45'
center_point_csv_path = '../c0/center_radii.csv'

##### data_preprocess.get_train_test_image_label
data_train_image_from_path = save_path_train_image
data_train_label_from_path = save_path_train_label
data_train_image_path = '../../data/result_1/train_image'
data_train_label_path = '../../data/result_1/train_label'
data_train_start_num = 1
data_train_end_num = 6
data_test_image_path = '../../data/result_1/test_image'
data_test_label_path = '../../data/result_1/test_label'
data_test_start_num = 6
data_test_end_num = 46

################################################
#### main.py

# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
# # data_gen_args = dict()

batch_size = 16
train_path = '../../data/result_1'
classes_image = 'train_image'
classes_label = 'train_label'
flag_multi_class = True
num_class = 4
target_size = (256, 256)
# data_gen_save_to_dir = '../data/result/aug'
data_gen_save_to_dir = None

predict_nii_gz_result_path = '../../data/result_1/predict_nii_gz_result/'
save_path = '../../data/result_1/test_label_for_predict/'
save_model_name = 'p3d_100_30.h5'
steps_per_epoch = 100
epochs = 30

test_path = data_test_image_path
test_image_nums = 800

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
input_size = (20, 256, 256, 1)
