from ner_model import NerModel

model = NerModel("data/ner_dataset_2.csv")
model.fit()
model.save("trained/train_2-20.pkl")
