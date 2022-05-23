import re

if __name__ == "__main__":
    training_set_raw = [re.sub(r'[^a-zA-Z0-9 ]', '',(line.lower())) for line in open("./trainingSet.txt")]

    training_set_token = [line.split() for line in training_set_raw]

    test_set_raw = [re.sub(r'[^a-zA-Z0-9 ]', '',(line.lower())) for line in open("./testSet.txt")]

    test_set_token = [line.split() for line in test_set_raw]

    

