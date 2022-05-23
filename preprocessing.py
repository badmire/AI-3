import re

if __name__ == "__main__":
    training_set_raw = [re.sub(r'[^a-zA-Z0-9 ]', '',(line.lower())) for line in open("./trainingSet.txt")]

    training_set_token = [line.split() for line in training_set_raw]

    test_set_raw = [re.sub(r'[^a-zA-Z0-9 ]', '',(line.lower())) for line in open("./testSet.txt")]

    test_set_token = [line.split() for line in test_set_raw]

    vocab = dict()

    for sentence in training_set_token:
        if sentence[-1]:
            for word in sentence[:-1]:
                if word in vocab:
                    vocab[word][-1] = 1
                else:
                    vocab[word] = [0,1]
        else:
                if word in vocab:
                    vocab[word][0] = 1
                else:
                    vocab[word] = [1,0]

    sorted(vocab)

    for word in vocab:
        if vocab[word] == [1,1]:
            print(word,vocab[word])

