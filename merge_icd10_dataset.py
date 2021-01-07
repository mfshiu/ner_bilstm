import pandas as pd
import re

def read_data_a(xlsx_path):
    data = []
    xlsx = pd.read_excel(xlsx_path, engine='openpyxl')
    for i in xlsx.index:
        print("\rRead %s, Line: %d" % (xlsx_path, i), end="")
        a = {
            "id": xlsx.ID加密[i],
            "birthday": str(xlsx.YYMMDD[i]),
            "admission_date": xlsx.入院日期[i].strftime("%Y%m%d"),
            "chief_compliant": trim_ingores(xlsx.主訴[i]),
            "history": trim_ingores(xlsx.病史[i]),
            "pathology": trim_ingores(xlsx.values[i][13]),  # 理學檢查發現
            "exam": trim_ingores(xlsx.檢驗[i]),
            "others": "%s|%s|%s|%s" % (trim_ingores(xlsx.values[i][17]),    # 病理報告
                                       trim_ingores(xlsx.values[i][18]),    # 手術日期及方法
                                       trim_ingores(xlsx.values[i][19]),    # 住院治療經過
                                       trim_ingores(xlsx.values[i][20]))    # 合併症與病發症
        }
        data.append(a)
    print("\rRead %s, Total: %d lines, done." % (xlsx_path, i + 1))

    return data


def read_data_b(xlsx_path):
    data = {}
    for sheet in range(3):
        xlsx = pd.read_excel(xlsx_path, sheet_name=sheet, engine='openpyxl')
        for i in xlsx.index:
            print("\rRead %s, sheet: %d, Line: %d" % (xlsx_path, sheet, i), end="")
            key = ",".join([xlsx.ID加密[i], str(xlsx.YYMMDD[i]), xlsx.入院日期[i].strftime("%Y%m%d")])
            if key not in data:
                data[key] = []
            data[key].append(xlsx.診斷代號[i])
        print("\rRead %s, sheet: %d, Total: %d lines, done." % (xlsx_path, sheet, i + 1))

    return data


def trim_ingores(text):
    return re.sub('[\n\t"]', ' ', text)
    # return text.replace("\n", " ").replace("\t", " ")


if __name__ == '__main__':
    # input_path_a = "data/icd10_dataset_a_1000.xlsx"
    # input_path_b = "data/icd10_dataset_b_3000.xlsx"
    # output_path = "data/icd10_dataset_ab_1000.tsv"
    input_path_a = "data/icd10_dataset_a.xlsx"
    input_path_b = "data/icd10_dataset_b.xlsx"
    output_path = "data/icd10_dataset_ab.tsv"

    data_a = read_data_a(input_path_a)
    data_b = read_data_b(input_path_b)

    lines = []
    for a in data_a:
        key = ",".join([a["id"], a["birthday"], a["admission_date"]])
        print("\r[%d] Merge %s" % (len(lines), key), end="")
        if key in data_b:
            line = "%s\t%s\t%s\t%s\t%s\t%s\n" \
                    % (",".join(data_b[key]),
                       a["chief_compliant"], a["history"], a["pathology"], a["exam"], a["others"])
            lines.append(line)
    print("\rMerge done. Total: %d" % (len(lines),))

    print("Write to: %s" % (output_path,))
    with open(output_path, "w") as fp:
        fp.write("icd10\tchief_compliant\thistory\tpathology\texam\tothers\n")
        fp.writelines(lines)

    print("Completed.")
