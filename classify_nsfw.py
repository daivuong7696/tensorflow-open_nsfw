#!/usr/bin/env python
import sys
import os
import argparse
import tensorflow as tf
from shutil import copyfile

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader

import numpy as np


IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

def classify_to_folder(args, images_names, images_scores):
    for i in range(len(images_names)):
        source = os.path.join(args.input_file, images_names[i])
        dest = os.path.join(args.input_file, str(round(images_scores[i][1], 1)))
        if not os.path.isdir(dest):
            os.mkdir(dest)
        copyfile(source, os.path.join(dest, images_names[i]))


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", help="Path to the input image.\
                        Only jpeg images are supported.")
    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")

    parser.add_argument("-l", "--image_loader",
                        default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument("-t", "--input_type",
                        default=InputType.TENSOR.name.lower(),
                        help="input type",
                        choices=[InputType.TENSOR.name.lower(),
                                 InputType.BASE64_JPEG.name.lower()])

    args = parser.parse_args()

    model = OpenNsfwModel()

    with tf.Session() as sess:

        input_type = InputType[args.input_type.upper()]
        model.build(weights_path=args.model_weights, input_type=input_type)

        fn_load_image = None

        if input_type == InputType.TENSOR:
            if args.image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(sess)
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        sess.run(tf.global_variables_initializer())
        images = []
        images_names = []
        for i in os.listdir(args.input_file):
            images_names.append(i)
            image_path = os.path.join(args.input_file, i)
            image = fn_load_image(image_path)
            if images == []:
                images = image
                print(image_path)
            else:
                images = np.concatenate((images, image), axis=0)
        image = images

        predictions = \
            sess.run(model.predictions,
                     feed_dict={model.input: image})

        classify_to_folder(args, images_names, predictions)
        #for i in range(len(images_names)):
        #    print("Results for '{}'".format(images_names[i]))
        #    print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[i]))

if __name__ == "__main__":
    main(sys.argv)
