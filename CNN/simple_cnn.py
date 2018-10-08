#coding=utf-8
import numpy
import sys
"""
Implement a simple cnn using numpy
"""

class SimpleCNN(object):
    def __init__(self):
        return

    def __conv_layer(self, img, cur_filter, stride = 1):
        """
        Calcualte the convoltion result using current filter. It's implemented with no stride.

        Todo: add the stride

        Args:
            img: origin img, matrix
            cur_filter: current filter, matrix

        Output:
            Convolution matrix
        """
        filter_size = cur_filter.shape[0]
        rows = img.shape[0]
        cols = img.shape[1]
        result = numpy.zeros((rows - filter_size + 1, cols - filter_size + 1))
        for i in numpy.arange(0, rows - filter_size + 1, stride):
            for j in numpy.arange(0, cols - filter_size + 1, stride):
                cur_img = img[i : i + filter_size, j : j + filter_size]
                cur_result = cur_img * cur_filter
                conv_sum = numpy.sum(cur_result)
                result[i, j] = conv_sum
        return result


    def conv(self, img, conv_filter, stride = 1):
        """
        Convolution layer, calculate the convolution of the img. When conv_filter shape is 3, each element means filter size, filter shape.

        Args:
            img: matrix, image or word sequence
            conv_filter: feature map
        """
        if len(img.shape) > 2 or len(conv_filter.shape) > 3: # Check if number of image channels matches the filter depth.
            if img.shape[-1] != conv_filter.shape[-1]:
                raise ValueError("Error: Number of channels in both image and filter must match.")
                sys.exit()
        if conv_filter.shape[1] != conv_filter.shape[2]:
            raise ValueError("The filter must be a square matrix, the number of rows and number of cols must equal!")
        if conv_filter.shape[1] % 2 == 0:
            raise ValueError("Division of the filter must be odd number")
        feature_map = numpy.zeros((img.shape[0]-conv_filter.shape[1]+1,
                                img.shape[1]-conv_filter.shape[1]+1,
                                conv_filter.shape[0]))

        for filter_num in xrange(conv_filter.shape[0]):
            print ("Feature map", filter_num)
            cur_filter = conv_filter[filter_num,:] # Get the current filter
            if len(cur_filter.shape) > 2:
                conv_map = self.__conv_layer(img[:, :, 0], cur_filter[:, :, 0], stride)
                for cur_num in xrange(1, cur_filter.shape[-1]):
                    conv_map += self.__conv_layer(img[:, :, cur_num], cur_filter[:, :, cur_num], stride)
            else:
                conv_map = self.__conv_layer(img, cur_filter, stride)
            feature_map[:, :, filter_num] = conv_map
        return feature_map

    def __max_pooling(self, feature_map, size, stride):
        """
        Max pooling.
        """
        pool_out = numpy.zeros(((feature_map.shape[0] - size + 1) / stride + 1,(feature_map.shape[1] - size + 1) / stride + 1,feature_map.shape[-1]))
        for map_num in xrange(feature_map.shape[-1]):
            r2 = 0
            for i in numpy.arange(0, feature_map.shape[0] - size + 1, stride):
                c2 = 0
                for j in numpy.arange(0, feature_map.shape[1] - size + 1, stride):
                    pool_out[r2, c2, map_num] = numpy.max([feature_map[i : i + size,  j : j + size]])
                    c2 += 1
                r2 += 1
        return pool_out

    def __avg_pooling(self, feature_map, size, stride):
        """
        Average pooling
        """
        pool_out = numpy.zeros(((feature_map.shape[0] - size + 1) / stride + 1,(feature_map.shape[1] - size + 1) / stride + 1,feature_map.shape[-1]))
        for map_num in xrange(feature_map.shape[-1]):
            r2 = 0
            for i in numpy.arange(0, feature_map.shape[0] - size + 1, stride):
                c2 = 0
                for j in numpy.arange(0, feature_map.shape[1] - size + 1, stride):
                    pool_out[r2, c2, map_num] = numpy.average([feature_map[i : i + size,  j : j + size]])
                    c2 += 1
                r2 += 1
        return pool_out

    def pooling(self, feature_map, size = 2, stride = 2, options = 1):
        """
        Pooling layer, with two options, max pooling or average pooling.

        Args:
            feature_map: matrix
            size: pool size
            stride: sliding windows size
            options: 1 max pooling, 2 average pooling
        """
        if options == 1:
            print "Options is 1, process max pooling"
            return self.__max_pooling(feature_map, size, stride)
        elif options == 2:
            print "Options is 2, process average pooling"
            return self.__avg_pooling(feature_map, size, stride)
        else:
            raise ValueError("Invalid options, optional value is 1 and 2. 1 for max pooling, 2 for average pooling.")

    def relu(self, feature_map):
        """
        Relu activation function

        """
        relu_out = numpy.zeros(feature_map.shape)
        for map_num in range(feature_map.shape[-1]):
            for r in numpy.arange(0,feature_map.shape[0]):
                for c in numpy.arange(0, feature_map.shape[1]):
                    relu_out[r, c, map_num] = numpy.max([feature_map[r, c, map_num], 0])
        return relu_out