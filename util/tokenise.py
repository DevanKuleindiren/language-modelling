import argparse
import nltk.data
import nltk.tokenize

parser = argparse.ArgumentParser(description="Tokenise a given file.")
parser.add_argument("--input_file_path", dest="input_file_path", help="The input file to tokenise.")
parser.add_argument("--output_file_path", dest="output_file_path", help="The output file for the tokenised text.")
parser.add_argument("--tweets", action="store_true" ,help="Set if the input file contains tweets.")

def main():
    args = parser.parse_args()

    with open(args.input_file_path) as f_in, open(args.output_file_path, "w") as f_out:
        for s in f_in:
            s_encoded_lowercase = s.decode("utf-8").lower()
            if args.tweets:
                line = nltk.tokenize.TweetTokenizer().tokenize(s_encoded_lowercase)
            else:
                line = nltk.tokenize.word_tokenize(s_encoded_lowercase)
            for w in line:
                f_out.write(w.encode("utf-8"))
                f_out.write(" ")
            f_out.write("\n")

if __name__ == "__main__":
    main()
