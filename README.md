# Language Modelling

This is the source code for my part II project on language modelling.

## Dependencies

In order to run the code for this project, you will need to install Bazel 0.3.2, which can be found
[here](https://github.com/bazelbuild/bazel/releases/tag/0.3.2).

## Set Up
In your desired directory:

1. `git clone -b v0.11.0_fix https://github.com/DevanKuleindiren/tensorflow.git`
2. `cd tensorflow`
3. `./configure` and follow the instructions.
4. `cd tensorflow`
5. `git clone https://github.com/DevanKuleindiren/language-modelling.git`
6. `mv language-modelling Source`
7. `cd Source`

## Running the Code

The code in this project is designed to be run using [Bazel](https://bazel.build). The project is split up into a
hierarchy of targets. From within the Source directory, running a target named `a/b:c` can be done as follows

    ```
    bazel run a/b:c -- ARGS
    ```

where `ARGS` is replaced with the relevant arguments for the program. The main targets in this project are:

* `:main` - the main command line interface to running inference on trained language models.
* `lm/ngram:train` - for training n-gram language models.
* `lm/rnn:rnn` - for training RNN-based language models.
* `benchmark:main` - for benchmarking the language models.
