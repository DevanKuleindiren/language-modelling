import argparse
import nltk.data
import nltk.tokenize

parser = argparse.ArgumentParser(description="Tokenise a given file.")
parser.add_argument("--input_file_path", dest="input_file_path", help="The input file to tokenise.")
parser.add_argument("--output_file_path", dest="output_file_path", help="The output file for the tokenised text.")
parser.add_argument("--tweets", action="store_true" ,help="Set if the input file contains tweets.")
parser.add_argument("--new_lines", action="store_true" ,help="Set if the input should be split into one sentence per line.")

def main():
    args = parser.parse_args()

    with open(args.input_file_path) as f:
        # Read the file's text into a list of strings. If new_lines is set, then each string will be a sentence.
        # Otherwise, there will just be one string representing the whole file.
        if args.new_lines:
            text = [f.read().decode("utf-8")]
        else:
            tokeniser = nltk.data.load('tokenizers/punkt/english.pickle')
            text = tokeniser.tokenize(f.read().decode("utf-8"))

    with open(args.output_file_path, "w") as f:
        for s in text:
            if args.tweets:
                line = nltk.tokenize.TweetTokenizer().tokenize(s)
            else:
                line = nltk.tokenize.word_tokenize(s)
            for w in line:
                f.write(w.encode("utf-8"))
                f.write(" ")
            f.write("\n")

if __name__ == "__main__":
    main()
