import matplotlib.pyplot as plt
import torch

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5, save_path=None):

    figsize = (num_cols * scale, num_rows*scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i , (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.clf()
        plt.close()
