import random
import re


MAX_LENS = 70
tags_history = dict()
coerce_words = dict([(k, 1) for k in ["，", "個管師"]])


def add_tag_history(tag, name):
    if tag not in tags_history:
        tags_history[tag] = list()
    tags_history[tag].append(name)


def parse_input(input_path):
    sentences = []

    with open(input_path, 'r', encoding='utf8') as fp:
        lines = fp.readlines()

    for line in lines[1:]:
        cells = line.split("\t")
        icd10s = cells[0].split(",")
        text = cells[1]
        sens = re.split('; |, |\. |! ', text)
        for sen in sens:
            words = sen.split(" ")
            ww = []
            for w in words:
                w = w.replace(",", " ").strip()
                if len(w):
                    ww.append((w, random.choice(icd10s)))
            if len(ww):
                sentences.append(ww)

    return sentences


def gen_output_rows(sentences, start_id, show_head):
    rows = []
    if show_head:
        rows.append("Sentence #,Word,POS,Tag")
    id = start_id
    for sen in sentences:
        rows.append("%s,%s,_,%s"%("Sentence: " + str(id), sen[0][0], sen[0][1]))
        for c in sen[1:]:
            rows.append(",%s,_,%s" % (c[0], c[1]))
        id += 1

    return rows


if __name__ == '__main__':
    input_path = "data/icd10_dataset_ab_1000.tsv"
    output_path = "data/icd10_dataset.csv"

    print("\n\n##### START CONVERT #####")
    print("input_path: " + input_path)
    print("output_path: " + output_path)
    print("#########################\n")

    sentences = parse_input(input_path)
    lines = [line + "\n" for line in gen_output_rows(sentences, 1, True)]
    with open(output_path, "w") as fp:
        fp.writelines(lines)

    print("Completed.")
