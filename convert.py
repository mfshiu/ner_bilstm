# 1
import random
from ckiptagger import construct_dictionary, WS
import copy


MAX_LENS = 70
ALTER_SIZE = 64
tags_history = dict()
coerce_words = dict([(k, 1) for k in ["，", "個管師"]])


def add_tag_history(tag, name):
    if tag not in tags_history:
        tags_history[tag] = list()
    tags_history[tag].append(name)


def extend_sentence(sentence, extend_count):
    def get_random_word(tag, default_word):
        word = default_word
        if tag in tags_history:
            words = tags_history[tag]
            if len(words) > 0:
                word = random.choice(words)
        return word

    extends = [sentence]
    sentences = [sentence.copy() for i in range(extend_count - 1)]
    for sen in sentences:
        change_count = 0
        for i, word in enumerate(sen):
            if word[1][:1] == "B":
                new_word = get_random_word(word[1], word[0])
                if new_word != word[0]:
                    sen[i] = (get_random_word(word[1], word[0]), word[1])
                    change_count += 1
        if change_count > 0:
            extends.append(sen)

    return extends


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


def get_sentences_to_alter(sentences):
    sentences2 = []

    def count_O(sen):
        cnt = 0
        for w in sen:
            if w[1] == "O":
                cnt += 1
        return cnt

    for i, sen in enumerate(sentences):
        if count_O(sen) < len(sen):
            sentences2.append(sen)

    return sentences2


def alter_sentence(sentences):
    sentences2 = []

    for i, sen in enumerate(sentences):
        sen2 = sen.copy()
        for j, w in enumerate(sen2):
            if w[1] != "O":
                w2 = random.choice(tags_history[w[1]])
                sen2[j] = (w2, w[1])
        sentences2.append(sen2)

    return sentences2

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
    input_path = "data/train_3.txt"
    output_path = "data/ner_dataset_3-64d.csv"

    print("\n\n##### START CONVERT #####")
    print("input_path: " + input_path)
    print("output_path: " + output_path)
    print("ALTER_SIZE: " + str(ALTER_SIZE))
    print("#########################\n")

    sentences = parse_input(input_path)
    lines = [line + "\n" for line in gen_output_rows(sentences, 1, True)]
    sentences_to_alter = get_sentences_to_alter(sentences)

    with open(output_path, "w") as fp:
        fp.writelines(lines)
        start_index = len(sentences) + 1
        for i in range(ALTER_SIZE):
            sentences2 = alter_sentence(sentences_to_alter)
            lines = [line + "\n" for line in gen_output_rows(sentences2, start_index, False)]
            fp.writelines(lines)
            start_index += len(sentences2)

    print("Completed.")
