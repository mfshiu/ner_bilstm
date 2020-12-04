# 1
import random
from ckiptagger import construct_dictionary, WS


MAX_LENS = 70
tags_history = dict()


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
    coerce_words = dict([(k, 1) for k in ["，"]])
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
            sen.append((w, tags[i]))
            if tags[i][:1] == "B":
                add_tag_history(tags[i], w)
            if w == "。" or w == "？":
                # sentences.append(sen)
                sens = extend_sentence(sen, 4)
                for s in sens:
                    sentences.append(s)
                sen = []
            i += len(w)

        # # Generate answer sentence
        # for row in rows[2:]:
        #     tokens = row.split('\t')
        #     tag = "B-" + tokens[4][:3]
        #     sentences.append([(tokens[3], tag)])

    return sentences


def gen_output_rows(sentences):
    rows = ["Sentence #,Word,POS,Tag"]
    for i, sen in enumerate(sentences):
        rows.append("%s,%s,_,%s"%("Sentence: " + str(i + 1), sen[0][0], sen[0][1]))
        for c in sen[1:]:
            rows.append(",%s,_,%s" % (c[0], c[1]))

    return rows


if __name__ == '__main__':
    input_path = "data/train_2.txt"
    output_path = "data/ner_dataset_2-4d.csv"

    sentences = parse_input(input_path)
    lines = gen_output_rows(sentences)
    lines = [line + "\n" for line in lines]
    with open(output_path, "w") as fp:
        fp.writelines(lines)

    print("Completed.")
