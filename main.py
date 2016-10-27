import lm.ngram.smoothing.add_one as add_one

def main():
    trigram = add_one.AddOne(3)
    trigram.parse_file("/Users/devankuleindiren/Desktop/test.txt")

    print(trigram._trie)

    while True:
        x = input("Sequence  : ")
        word_seq = x.split(" ")
        print("Prediction: %s" % trigram.predict(word_seq))

if __name__ == "__main__":
    main()
