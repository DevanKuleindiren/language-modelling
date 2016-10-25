import lm.ngram.model as ngram

def main():
    trigram = ngram.NGramLM(3)
    trigram.parse_file("/Users/devankuleindiren/Desktop/test.txt")

    while True:
        x = input("Sequence  : ")
        word_seq = x.split(" ")
        print("Prediction: %s" % trigram.predict(word_seq))

if __name__ == "__main__":
    main()
