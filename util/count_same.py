import argparse

parser = argparse.ArgumentParser(description="Compute the percentage of matching sentences.")
parser.add_argument("--error_free_input_path", dest="error_free_input_path", help="The error free input file path.")
parser.add_argument("--error_prone_input_path", dest="error_prone_input_path", help="The error prone input file path.")

window_size = 5

def main():
    args = parser.parse_args()

    count = 0.0
    count_same = 0.0

    with open(args.error_free_input_path) as error_free_in, open(args.error_prone_input_path) as error_prone_in:
        for l1, l2 in zip(error_free_in, error_prone_in):
            if l1 == l2:
                count_same = count_same + 1
            count = count + 1

    print "%d%% of the sentences match." % ((count_same / count) * 100)


if __name__ == "__main__":
    main()
