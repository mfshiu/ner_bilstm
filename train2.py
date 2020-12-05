from ner_model import NerModel
import os

# import keras.backend as K
# cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
# K.set_session(K.tf.Session(config=cfg))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data_path = "data/ner_dataset_2-8d.csv"
model_path = "trained/train_2-40-80-8d.pkl"

print("\n\n##### START #####")
print("data_path: " + data_path)
print("model_path: " + model_path)

model = NerModel(data_path, embedding_size=80)
model.fit(epochs=40)
model.save(model_path)
