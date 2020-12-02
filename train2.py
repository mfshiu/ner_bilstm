from ner_model import NerModel
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

model = NerModel("data/ner_dataset_2.csv", embedding_size=80)
model.fit(epochs=40)
model.save("trained/train_2-40-80.pkl")
