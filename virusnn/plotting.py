import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')


def plot_sequence_heatmap(data, filename=None, title=None, x_label="Positions", y_label="Nucleotides", y_ticks=None):
    """ Plots heatmap of sequence matrix (rows corresponds nucleotides, columns correspond to genomic location). """
    if y_ticks is None:
        y_ticks = ['A', 'C', 'G', 'T', '-', 'N']

    plt.figure(figsize=(20, 5))
    plt.imshow(data, interpolation="nearest", cmap='Blues')
    plt.xticks([], [])
    plt.xlabel(x_label)
    plt.yticks(np.arange(data.shape[0]), y_ticks[0:data.shape[0]])
    plt.ylabel(y_label)
    plt.ylim([data.shape[0] - 0.5, -0.5])
    #plt.colorbar()
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == '__main__':
    print("Creating dummy sequence matrix...")
    data = np.random.randint(0, 10, size=(6, 50))

    # Display the marginal one-hot sequence matrix
    plot_sequence_heatmap(data, title="Example Heatmap")

    # Save the marginal one-hot sequence matrix
    print("Saving the matrix heatmap...")
    plot_sequence_heatmap(data, filename="plots/heatmap_example_dummy.png", title="Example Heatmap")


