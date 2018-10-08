#coding=utf-8
import matplotlib
import numpy
import skimage.data

from simple_cnn import SimpleCNN

"""
Testing simple cnm
"""
def show_originale_img(img):
    """
    Show original image, using matplotlib
    """
    fig0, ax0 = matplotlib.pyplot.subplots(nrows=1, ncols=1)
    ax0.imshow(img).set_cmap("gray")
    ax0.set_title("Original img")
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    matplotlib.pyplot.savefig("in_img.png", bbox_inches="tight")
    matplotlib.pyplot.close(fig0)

def show_layeroutput(layers_out, nrows, ncols, titles, img_name):
    """
    Each layer has convolution output, relu output and pooling output, show these results
    """
    if len(layers_out) != nrows:
        raise ValueError("Input layerout does not match rows")
    fig, ax = matplotlib.pyplot.subplots(nrows = nrows, ncols = ncols)
    for i in xrange(nrows):
        for j in xrange(ncols):
            if layers_out[i].shape[-1] != ncols:
                raise ValueError("Input layerout does not match cols")
                exit(1)
            img = layers_out[i][:, :, j]
            ax[i, j].imshow(img).set_cmap("gray")
            ax[i, j].set_title(titles[i])
            ax[i, j].get_xaxis().set_ticks([])
            ax[i, j].get_yaxis().set_ticks([])
    matplotlib.pyplot.savefig(img_name, bbox_inches="tight")
    matplotlib.pyplot.close(fig)

def process_img():
    simplecnn = SimpleCNN()
    img = skimage.data.camera()
    img = skimage.color.rgb2gray(img)
    l1_filter = numpy.zeros((2, 3, 3))
    l1_filter[0, :, :] = numpy.array([[[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]]])
    l1_filter[1, :, :] = numpy.array([[[1,   1,  1],
                                     [0,   0,  0],
                                   [-1, -1, -1]]])
    print "Conv layer 1..."
    l1_feature_map = simplecnn.conv(img, l1_filter)
    print "Relu..."
    l1_feature_map_relu = simplecnn.relu(l1_feature_map)
    print "Max pooling..."
    l1_filter_map_relu_pooling = simplecnn.pooling(l1_feature_map_relu, options = 2)
    print "End with layer 1..."

    print "Conv layer 2..."
    l2_filter = numpy.random.rand(3, 5, 5, l1_filter_map_relu_pooling.shape[-1])
    l2_feature_map = simplecnn.conv(l1_filter_map_relu_pooling, l2_filter, 2)
    print "Relu..."
    l2_feature_map_relu = simplecnn.relu(l2_feature_map)
    print "Max pooling..."
    l2_feature_map_relu_pooling = simplecnn.pooling(l2_feature_map_relu, options = 2)
    print "End wirh layer 2..."

    # Reveal the image
    show_originale_img(img)
    layers_out = [l1_feature_map, l1_feature_map_relu, l1_filter_map_relu_pooling]
    nrows = 3
    ncols = l1_feature_map.shape[-1]
    titles = ["L1 map", "L1 map relu", "L1 map relu pool"]
    img_name = "L1.png"
    show_layeroutput(layers_out, nrows, ncols, titles, img_name)

    layers_out = []
    layers_out = [l2_feature_map, l2_feature_map_relu, l2_feature_map_relu_pooling]
    nrows = 3
    ncols = l2_feature_map.shape[-1]
    titles = ["L2 map", "L2 map relu", "L2 map relu pool"]
    img_name = "L2.png"
    show_layeroutput(layers_out, nrows, ncols, titles, img_name)


if __name__ == "__main__":
    process_img()