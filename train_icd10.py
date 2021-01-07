from icd10_model import NerModel
import os

# import keras.backend as K
# cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
# K.set_session(K.tf.Session(config=cfg))

# embedding_size = 20
embedding_size = 20
epochs = 10
data_path = "data/icd10_dataset.csv"
model_path = "trained/weight-e10-h20-v20.pkl"

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("\n\n##### START TRAINING #####")
    print("data_path: " + data_path)
    print("model_path: " + model_path)
    print("##############################\n")

    model = NerModel(data_path, embedding_size=embedding_size)
    model.fit(epochs=epochs)
    model.save(model_path)
