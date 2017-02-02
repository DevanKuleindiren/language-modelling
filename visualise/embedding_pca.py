import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from google.protobuf import text_format
from sklearn.decomposition import PCA
from tensorflow.Source.lm import vocab_pb2

tf.flags.DEFINE_string("model_path", None, "The file path of the RNN model.")
tf.flags.DEFINE_string("embedding_tensor_name", "rnn/embedding:0", "The name of the embedding Tensor.")

FLAGS = tf.flags.FLAGS

def main():
    if not FLAGS.model_path:
        raise ValueError("Must set --model_path.")

    # Load the model vocab.
    id_to_word = {}
    vocab = vocab_pb2.VocabProto()
    with open(os.path.join(FLAGS.model_path, "vocab.pbtxt"), "rb") as f:
        text_format.Merge(f.read(), vocab)
        for i in vocab.item:
            id_to_word[i.id] = i.word

    # Load the model graph into a TensorFlow session and evaluate the embeddings tensor.
    graph_def = tf.GraphDef()
    with open(os.path.join(FLAGS.model_path, "graph.pb"), "rb") as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")
    with tf.Session() as sess:
        embeddings = np.array(sess.run(FLAGS.embedding_tensor_name))

    # Reduce the dimensionality of the embeddings tensor to 2D.
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    embeddings_reduced = pca.transform(embeddings)

    # Plot and label the points.
    fig, ax = plt.subplots()
    ax.scatter(embeddings_reduced[:,0], embeddings_reduced[:,1])
    for i in sorted(id_to_word.keys()):
        ax.annotate(id_to_word[i], (embeddings_reduced[:,0][i], embeddings_reduced[:,1][i]))
    plt.show()


if __name__ == "__main__":
    main()
