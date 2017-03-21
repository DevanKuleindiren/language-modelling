import argparse

parser = argparse.ArgumentParser(description="Compute the percentage of matching sentences.")
parser.add_argument("--error_prone_input_path", dest="error_prone_input_path", help="The error prone input file path.")
parser.add_argument("--error_free_input_path", dest="error_free_input_path", help="The error free input file path.")


def main():
    args = parser.parse_args()

    line_number = 1
    with open(args.error_prone_input_path) as error_prone_in, open(args.error_free_input_path) as error_free_in:
        for l1, l2 in zip(error_prone_in, error_free_in):
            print "Line %d." % line_number
            for w1, w2 in zip(l1.split(), l2.split()):
                if w1 != w2:
                    print "CORRECTION: %s --> %s" % (w1, w2)
            line_number += 1


if __name__ == "__main__":
    main()
