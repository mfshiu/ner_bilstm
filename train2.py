from ner_model import NerModel

model = NerModel("data/ner_dataset_2.csv", embedding_size=80)
model.fit(epochs=40)
model.save("trained/train_2-40.pkl")
