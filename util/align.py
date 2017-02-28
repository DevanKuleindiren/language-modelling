import argparse

parser = argparse.ArgumentParser(description="Tokenise a given file.")
parser.add_argument("--error_free_input", dest="error_free_input", help="The file path of the error free input.")
parser.add_argument("--error_prone_input", dest="error_prone_input", help="The file path of the error prone input.")
parser.add_argument("--error_free_output", dest="error_free_output", help="The file path of the error free output.")
parser.add_argument("--error_prone_output", dest="error_prone_output", help="The file path of the error prone output.")

def main():
    args = parser.parse_args()

    with open(args.error_free_input) as free_in, \
         open(args.error_prone_input) as prone_in, \
         open(args.error_free_output, "w") as free_out, \
         open(args.error_prone_output, "w") as prone_out:
        for free_s, prone_s in zip(free_in, prone_in):
            if len(free_s.split()) == len(prone_s.split()):
                d = 0
                for w1, w2 in zip(free_s.split(), prone_s.split()):
                    if w1 != w2:
                        d += 1
                if d <= 2:
                    free_out.write(free_s)
                    prone_out.write(prone_s)

if __name__ == "__main__":
    main()
