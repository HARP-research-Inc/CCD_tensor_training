import json

if __name__ == "__main__":
    file_in = open("../data/textblock_tiny.json")
    file_out = open("../data/tiny_sentences.txt", 'w')    
    data = json.load(file_in)

    for verb in data:
        nouns = data[verb]

        subjects = nouns[0]
        objects = nouns[1]

        for subject in subjects:
            for object in objects:
                file_out.write(subject + " " + verb + " " + object + "\n")

        