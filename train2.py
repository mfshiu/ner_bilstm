from ner_model import NerModel
import os

# import keras.backend as K
# cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
# K.set_session(K.tf.Session(config=cfg))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = NerModel("data/ner_dataset_2-8d.csv", embedding_size=80)
model.fit(epochs=40)
model.save("trained/train_2-40-80-8d.pkl")
