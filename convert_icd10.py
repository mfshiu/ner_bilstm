import random
import re
import merge_icd10_dataset


MAX_LENS = 70
tags_history = dict()
coerce_words = dict([(k, 1) for k in ["，", "個管師"]])


def add_tag_history(tag, name):
    if tag not in tags_history:
        tags_history[tag] = list()
    tags_history[tag].append(name)


def remove_ignore_chars(word):
    word = word.replace(",", " ").strip()
    return word


skip_words = set([
    "for", "with", "this", "has", "was", "is", "are", "of", "or", "to",
    "our", "at", "in", "he", "and", "not", "no", "nil", "without", "a",
    "on", "via", "about", "to", "the", "she", "also", "have", "that", "ago",
    "from", "recent", "ago", "data", "but", "were", "as", "&", "by", "be",
    "some", "ever", "his", "him", "do", "done", "nil.", "-", "we", "got",
])


def is_skip_word(word):
    return word.lower() in skip_words


def parse_input(input_path):
    sentences = []

    with open(input_path, 'r', encoding='utf8') as fp:
        lines = fp.readlines()

    for line in lines[1:]:
        cells = line.split("\t")
        icd10s = cells[0].split(",")
        for text in cells[1:]:
            sens = re.split('; |, |\. |! ', text)
            for sen in sens:
                words = sen.split(" ")
                ww = []
                for w in words:
                    w = w.replace(",", " ").strip()
                    if len(w):
                        if not is_skip_word(w):
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


input_path = merge_icd10_dataset.output_path
output_path = "data/icd10_dataset_200.csv"

if __name__ == '__main__':
    print("\n\n##### START CONVERT #####")
    print("input_path: " + input_path)
    print("output_path: " + output_path)
    print("#########################\n")

    sentences = parse_input(input_path)
    lines = [line + "\n" for line in gen_output_rows(sentences, 1, True)]
    with open(output_path, "w") as fp:
        fp.writelines(lines)

    print("Completed.")
