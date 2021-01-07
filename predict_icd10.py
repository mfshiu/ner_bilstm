import os
from icd10_model import NerModel
import random
import train_icd10
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

taged_words = {
}

# input_path = "data/development_2.txt"
input_path = "data/test_icd10.txt"
output_path = "output/icd10-test.tsv"
data_path = train_icd10.data_path
model_path = train_icd10.model_path

model = NerModel(data_path, embedding_size=train_icd10.embedding_size)
model.load_weights(model_path)

print("\n\n######################## START PREDICT ########################")
print("input_path: " + input_path)
print("output_path: " + output_path)
print("data_path: " + data_path)
print("model_path: " + model_path)
print("###############################################################\n")


def wrap_sentences(wordss):
    def warp_list(words, size):
        if len(words) <= size:
            return [words]
        new_words = []
        while len(words) > size:
            new_words.append(words[:size])
            words = words[size:]
        if len(words) > 0:
            new_words.append(words[:size])
        return new_words

    new_wordss = []
    for words in wordss:
        new_wordss.extend(warp_list(words, 75))
    return new_wordss


# sentences = [
#     ["醫師", "：", "賈伯斯", "的", "你", "看了", "嗎", "？"],
#     ["前天", "又", "有", "在", "發燒", "喔", "。"],
#     ["民眾", "：", "我", "住", "在", "士林", "。"],
# ]
with open(input_path, "r") as fp:
    articles = [a[:-1] for i, a in enumerate(fp) if (i - 1) % 5 == 0]


def calculate_icd10s(tags):
    tags.remove("PAD")
    ss = set(tags)
    if len(ss) > 0:
        cnt = random.randint(0, int(len(ss)/2)) + 1
        aa = random.sample(tags, cnt)
    else:
        aa = []
    return aa


all_words = []
for article_id, article in enumerate(articles):
    print("[%d/%d] %s..." % (article_id, len(articles)-1, article[:50]))
    sentences = re.split('; |, |\. |! ', article)

    word_sentence_list = []
    for sen in sentences:
        words = sen.split(" ")
        ww = []
        for w in words:
            w = w.replace(",", " ").strip()
            if len(w):
                ww.append(w)
        if len(ww):
            word_sentence_list.append(ww)

    wordss = wrap_sentences(word_sentence_list)
    tagss = model.predict(wordss)

    all_tags = []
    for i, words in enumerate(wordss):
        for j, word in enumerate(words):
            tag = model.idx2tag[tagss[i][j]]
            all_tags.append(tag)
    icd10s = calculate_icd10s(all_tags)
    all_words.append((article_id, ",".join(icd10s)))

print("Output to: " + output_path)
rows = ["article_id\tICD-10\n"]
for w in all_words:
    rows.append("%d\t%s\n" % (w[0], w[1]))
with open(output_path, "w") as fp:
    fp.writelines(rows)
