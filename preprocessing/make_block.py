if __name__ == "__main__":
    file = open("../data_raw/IMDB_Dataset.csv", 'r')
    file_out = open("../data_raw/IMDB_Textblock.txt", 'w')


    data = file.readlines()

    header = True

    for line in data:
        if header:
            header = False
            continue

        text = line.rsplit(',', 1)[0]
        text = text.strip("\"")
        text = text.strip("\n")
        #text = text.strip()
        text = text.replace("<br /><br />", "")

        # sentences = text.split(".")
        # for sentence in sentences:
        #     sentence.strip("\n")
        #     file_out.write(sentence + "\n")
        file_out.write(text)