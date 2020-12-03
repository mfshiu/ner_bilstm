import os
from ner_model import NerModel
from ckiptagger import construct_dictionary, WS, POS, NER

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tag2fullname = {
    "acc": "account",
    "bio": "biomarker",
    "cli": "clinical_event",
    "con": "contact",
    "edu": "education",
    "fam": "family",
    "bel": "belonging_mark",
    "oth": "others",
    "ID": "ID",
    "loc": "location",
    "mon": "money",
    "nam": "name",
    "non": "none",
    "org": "organization",
    "pro": "profession",
    "spe": "special_skills",
    "tim": "time",
    "uni": "unique_treatment",
    "med": "med_exam",
}

input_path = "data/development_2.txt"
output_path = "output/aicup-40.tsv"
model = NerModel("data/ner_dataset_2.csv", embedding_size=80)
model.load_weights("trained/train_2-40.pkl")

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

tagO = "O"
tagO_index = model.tag2idx[tagO]

def merge_ckip_ner(wordss, tagss, entity_sentence_list):
    ckip_tags_map = {
        # "cardinal": "B-med",
        # "person": "B-nam",
        "org": "B-org",
        "organization": "B-org",
        "gpe": "B-loc",
        "loc": "B-loc",
        "location": "B-loc",
        "event": "B-cli",
        # "date": "B-tim",
        # "time": "B-tim",
        "percent": "B-med",
        "money": "B-mon",
    }

    for i, words in enumerate(wordss):
        if len(entity_sentence_list[i]) > 0:
            sentence = "".join(words)
            ckip_tags = [None for a in range(len(sentence))]
            for es in entity_sentence_list[i]:
                if es[2].lower() in ckip_tags_map:
                    ckip_tags[es[0]] = ckip_tags_map[es[2].lower()]
            offset = 0
            for j, word in enumerate(words):
                if ckip_tags[offset]:
                    if tagss[i][j] == tagO_index or tagss[i][j] == 0:
                        tagss[i][j] = model.tag2idx[ckip_tags[offset]]
                offset += len(word)

    return tagss

# sentences = [
#     ["醫師", "：", "賈伯斯", "的", "你", "看了", "嗎", "？"],
#     ["前天", "又", "有", "在", "發燒", "喔", "。"],
#     ["民眾", "：", "我", "住", "在", "士林", "。"],
# ]
with open(input_path, "r") as fp:
    articles = [a[:-1] for i, a in enumerate(fp) if (i - 1) % 5 == 0]

ws = WS("./ckipdata")  # , disable_cuda=not GPU)
pos = POS("./ckipdata")
ner = NER("./ckipdata")
delimiters = {"，", "。", "：", "？", "！", "；", ",", ":", "?", "!", ";"}

all_words = []
for article_id, article in enumerate(articles):
    print("[%d/%d] %s..." % (article_id, len(articles)-1, article[:50]))
    article2 = article.replace("。", "。 ").replace("？", "？ ").replace("！", "！ ")
    article2 = article2.replace("阿", "_")
    sentences = article2.split(" ")  # re.split("。|？|！|\n", article)
    word_sentence_list = ws(sentences, segment_delimiter_set=delimiters)
    wordss = wrap_sentences(word_sentence_list)
    tagss = model.predict(wordss)
    pos_sentence_list = pos(wordss)
    entity_sentence_list = ner(wordss, pos_sentence_list)
    tagss = merge_ckip_ner(wordss, tagss, entity_sentence_list)

    offset = 0
    for i, words in enumerate(wordss):
        last_word, last_tag = "", ""
        for j, word in enumerate(words):
            if j >= len(tagss[i]):
                print("Warning, article: %d, sentence: %d too long, truncated. len(%d), %s"
                      % (article_id, i, len(sentences[i]), sentences[i]))
                break
            tag = model.idx2tag[tagss[i][j]]
            tag_head = tag[:2]
            if tag_head == "I-":
                last_word += word
                if not last_tag:  # Avoid tag without B-
                    last_tag = tag[2:]
            else:
                if last_word:
                    all_words.append((article_id,
                                      offset - len(last_word),
                                      offset,
                                      last_word,
                                      tag2fullname[last_tag]))
                    last_word, last_tag = "", ""
                if tag_head == "B-":
                    last_word = word
                    last_tag = tag[2:]
            offset += len(word)

        if last_word:
            all_words.append((article_id,
                              offset - len(last_word),
                              offset,
                              last_word,
                              tag2fullname[last_tag]))

print("Output to: " + output_path)
rows = ["article_id\tstart_position\tend_position\tentity_text\tentity_type\n"]
for w in all_words:
    rows.append("%d\t%d\t%d\t%s\t%s\n" % (w[0], w[1], w[2], w[3], w[4]))
with open(output_path, "w") as fp:
    fp.writelines(rows)
