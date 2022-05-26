import re
import math


if __name__ == "__main__":

    # Classification
    training_set_raw = [
        re.sub(r"[^a-zA-Z0-9 ]", "", (line.lower()))
        for line in open("./trainingSet.txt")
    ]

    training_set_token = [line.split() for line in training_set_raw]

    test_set_raw = [
        re.sub(r"[^a-zA-Z0-9 ]", "", (line.lower())) for line in open("./testSet.txt")
    ]

    test_set_token = [line.split() for line in test_set_raw]

    vocab = {"good":dict(),"bad":dict()}
    test_vocab = {"good":dict(),"bad":dict()}
    total_training_words = 0
    total_test_words = 0

    for sentence in training_set_token:
        if int(sentence[-1]):
            for word in sentence[:-1]:
                total_training_words += 1
                if word in vocab["good"]:
                    vocab["good"][word] += 1
                else:
                    vocab["good"][word] = 1
        else:
            for word in sentence[:-1]:
                total_training_words += 1
                if word in vocab["bad"]:
                    vocab["bad"][word] += 1
                else:
                    vocab["bad"][word] = 1

    for sentence in test_set_token:
        if int(sentence[-1]):
            for word in sentence[:-1]:
                total_test_words += 1
                if word in test_vocab["good"]:
                    test_vocab["good"][word] += 1
                else:
                    test_vocab["good"][word] = 1
        else:
            for word in sentence[:-1]:
                total_test_words += 1
                if word in test_vocab["bad"]:
                    test_vocab["bad"][word] += 1
                else:
                    test_vocab["bad"][word] = 1

    all_words = sorted(set(list(vocab["bad"].keys()) + list(vocab["good"].keys())))

    # Sort it
    vocab["good"] = {key:vocab["good"][key] for key in sorted(vocab["good"].keys())}
    vocab["bad"] = {key:vocab["bad"][key] for key in sorted(vocab["bad"].keys())}


    line_1 = ""
    line_2 = ""
    line_3 = ""

    for word in all_words:
        line_1 += word+","
        
        if word in vocab["bad"] or word in test_vocab["bad"]:
            line_2 += "1,"
        else:
            line_2 += "0,"

        if word in vocab["good"] or word in test_vocab["good"]:
            line_3 += "1,"
        else:
            line_3 += "0,"

    line_1 += "classlabel\n"
    line_2 += "0\n"
    line_3 += "1\n"

    target = open("./test_out","w")
    target.write(line_1)
    target.write(line_2)
    target.write(line_3)
    target.close()

    total_good_training_words = 0
    total_bad_training_words = 0
    for word in vocab["good"]:
        total_good_training_words += vocab["good"][word]
    
    for word in vocab["bad"]:
        total_bad_training_words += vocab["bad"][word]


    # Training

    # Probablity of class variable
    CV_total = (len(all_words))
    CV_true = len(vocab["good"])
    CV_false = len(vocab["bad"])
    pCV_false = total_bad_training_words/total_training_words
    pCV_true = total_good_training_words/total_training_words

    # Calculate probability of parameters


    predictions = []

    for sentence in test_set_token:
        current_good = 1
        current_bad = 1
        unknown = []
        for word in sentence[:-1]:
            if word in vocab["good"]:
                # probability of word given positive review
                current_good *= math.log((vocab["good"][word] / CV_true) / CV_true)
            else:
                unknown.append(word)

            if word in vocab["bad"]:
                # probability of word given negative review
                current_bad *= math.log((vocab["bad"][word] / CV_true) / CV_true)
            else:
                unknown.append(word)

                # fGf = (CV_total - vocab["bad"][word])/CV_false # false given false
                # fGt = (CV_total - vocab["bad"][word])/CV_true # false given true
                # tGf = (vocab["bad"][word])/CV_false # true given false
                # tGt = (vocab["bad"][word])/CV_true # true given true
                # current_good = current_good * (tGt + fGt)
                # current_bad = current_bad * (tGf + fGf)
        
        if len(unknown) == len(sentence[:-1]):
            predictions.append(-1)
        else:
            current_bad = current_bad*pCV_true
            current_good = current_good*pCV_true

            if current_good > current_bad:
                predictions.append(1)
            else:
                predictions.append(0)

    correct = 0
    wrong = 0

    for i in range(len(predictions)):
        if predictions[i] == -1:
            continue
        elif predictions[i] == int(test_set_token[i][-1]):
            correct += 1

    print(correct,len(test_set_token))
    print(correct/len(test_set_token))
