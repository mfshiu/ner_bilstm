from ner_model import NerModel
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras.backend as K
cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
K.set_session(K.tf.Session(config=cfg))

model = NerModel("data/ner_dataset_2.csv", embedding_size=80)
model.fit(epochs=40)
model.save("trained/train_2-40-80.pkl")
