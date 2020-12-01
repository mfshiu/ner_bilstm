from ckiptagger import construct_dictionary, WS


MAX_LENS = 70


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
        print("\n[%d/%d] %s..."%(block_num+1, len(blocks), article[0:50]))

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
            if w == "。":
                sentences.append(sen)
                sen = []
            i += len(w)

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
    output_path = "data/ner_dataset_2.csv"

    sentences = parse_input(input_path)
    lines = gen_output_rows(sentences)
    lines = [line + "\n" for line in lines]
    with open(output_path, "w") as fp:
        fp.writelines(lines)

    print("Completed.")
