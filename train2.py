from ner_model import NerModel
import os

# import keras.backend as K
# cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
# K.set_session(K.tf.Session(config=cfg))

# embedding_size = 20
embedding_size = 512
epochs = 20
data_path = "data/ner_dataset_3-64d.csv"
# model_path = "trained/train_2-2.pkl"
model_path = "trained/train_3-20-512.pkl"

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("\n\n##### START TRAINING #####")
    print("data_path: " + data_path)
    print("model_path: " + model_path)
    print("##############################\n")

    model = NerModel(data_path, embedding_size=embedding_size)
    model.fit(epochs=epochs)
    model.save(model_path)
