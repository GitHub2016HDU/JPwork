#!/usr/bin/evn python
# -*- coding : utf-8 -*-
import tensorflow as tf

#
# # # # 3D model
# from model import model
#
# if __name__  == '__main__':
#     batch_size = 32
#     num_batch = 100
#     learning_rate = 0.01
#     keep_prob = 0.5
#     path = '/data/LUNA2016/cubic_normalization_npy'
#     test_path = '/data/LUNA2016/cubic_normalization_test'
#
#     print(" beigin...")
#     model = model(learning_rate, keep_prob, batch_size, 40)
#     model.inference_op(path, test_path, 0, True)
#


# # # 2D model
from model2D import model2D

if __name__ == '__main__':
    batch_size = 32
    num_batch = 100
    learning_rate = 0.01
    keep_prob = 0.5
    path = '/data/LUNA2016/cubic_normalization_npy'
    test_path = '/data/LUNA2016/cubic_normalization_test'

    print(" beigin...")
    # model2d = model2D()
    model2d = model2D(batch_size        =batch_size,
                      learni_rate       =learning_rate,
                      keep_prob         =keep_prob,
                      epoch             =50,
                      input_size        =40
                    )
    model2d.run_benchmark()
