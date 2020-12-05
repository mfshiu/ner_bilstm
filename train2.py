from ner_model import NerModel
import os

# import keras.backend as K
# cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
# K.set_session(K.tf.Session(config=cfg))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    data_path = "data/ner_dataset_3-16d.csv"
    model_path = "trained/train_3-60-160.pkl"

    print("\n\n##### START #####")
    print("data_path: " + data_path)
    print("model_path: " + model_path)

    model = NerModel(data_path, embedding_size=160)
    model.fit(epochs=60)
    model.save(model_path)
