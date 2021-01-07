# 1
import random
from ckiptagger import construct_dictionary, WS
import copy


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
        file_text = fp.read().encode('utf-8').decode('utf-8-sig')
    print("Initial CKIP...")
    delimiters = {"，", "。", "：", "？", "！", "；", ",", ":", "?", "!", ";"}
    ws = WS("./ckipdata")  # , disable_cuda=not GPU)
    blocks = file_text.split('\n\n--------------------\n\n')[:-1]
    for block_num, block in enumerate(blocks):
        rows = block.split('\n')
        article = rows[0]
        print("\n[%d/%d] %s..." % (block_num+1, len(blocks), article[0:50]))

        # Load tags
        tags = ["O" for i in range(len(article))]
        for row in rows[2:]:
            tokens = row.split('\t')
            start, end = int(tokens[1]), int(tokens[2])
            tag = tokens[4][:3]
            tags[start] = "B-" + tag
            for i in range(start + 1, end):
                tags[i] = "I-" + tag

        # Segment article
        print("  segment...", end="")
        words = ws([article],
                   segment_delimiter_set=delimiters,
                   coerce_dictionary=construct_dictionary(coerce_words))[0]
        print("%s..." % (words[0:10]))

        # Generate sentence
        i, sen = 0, []
        for w in words:
            if tags[i] != "O":
                add_tag_history(tags[i], w)
            sen.append((w, tags[i]))
            if w == "。" or w == "？":
                sentences.append(sen)
                sen = []
            i += len(w)

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
    input_path = "data/train.txt"
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
