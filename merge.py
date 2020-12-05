def read_articles(file_path):
    with open(file_path, "r", encoding="utf-8") as fp:
        articles = [a for i, a in enumerate(fp) if (i - 1) % 5 == 0]
    return articles


def read_articles_predicts(file_path):
    with open(file_path, "r", encoding="utf-8") as fp:
        rows = fp.readlines()[1:]
    if rows[-1][-1] != "\n":
        rows[-1] += "\n"
    rows = [row.split("\t") for row in rows]

    articles_predicts = []
    predicts = []
    article_id = rows[0][0]
    for row in rows:
        id = row[0]
        line = "\t".join(row[1:])
        if id != article_id:
            articles_predicts.append(predicts)
            predicts = []
            article_id = id
        predicts.append(line)
    articles_predicts.append(predicts)

    return articles_predicts


if __name__ == '__main__':
    input_path = "data/train_2.txt"
    development_path = "data/development_2.txt"
    result_path = "output/aicup-40-std.tsv"
    output_path = "data/train_3.txt"

    articles = read_articles(development_path)
    articles_predicts = read_articles_predicts(result_path)

    rows = []
    for i, article in enumerate(articles):
        rows.append(article)
        rows.append("article_id	start_position	end_position	entity_text	entity_type\n")
        rows.extend(articles_predicts[i])
        rows.append("\n--------------------\n\n")

    with open(input_path, "r") as fp:
        input_content = fp.read()

    with open(output_path, "w") as fp:
        fp.write(input_content)
        fp.writelines(rows)

    print("Completed.")
