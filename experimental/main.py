import experimental.ngram_lm as ngram

def main():
    trigram = ngram.NGramLM(3)
    trigram.parse_file("/Users/devankuleindiren/Downloads/1-billion-word-language-modeling-benchmark-r13output"
                       "/training-monolingual.tokenized.shuffled/news.en-00001-of-00100")

    while True:
        x = input("Sequence  : ")
        word_seq = x.split(" ")
        print("Prediction: %s" % trigram.predict(word_seq))

if __name__ == "__main__":
    main()
