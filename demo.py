from matplotlib import pyplot as plt
import os
import argparse
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
from PIL import Image
from matplotlib.patches import Patch


from preprocess import makeImageBetter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()

parser.add_argument('--im_path', type=str, default='demo/45719584.jpg',
                    help='input image paths.')

# color map
floorplan_map = {
    0: [255, 255, 255],  # background
    1: [192, 192, 224],  # closet
    2: [192, 255, 255],  # bathroom/washroom
    3: [224, 255, 192],  # living_room/kitchen/dining room
    4: [255, 224, 128],  # bedroom
    5: [255, 160, 96],  # hall
    6: [255, 224, 224],  # balcony
    7: [255, 255, 255],  # not used
    8: [255, 255, 255],  # not used
    9: [255, 60, 128],  # door & window
    10: [0, 0, 0]  # wall
}

floorplan_legend = {
    'background': [255, 255, 255],
    'closet': [192, 192, 224],
    'bathroom/washroom': [192, 255, 255],
    'living_room/kitchen/dining room': [224, 255, 192],
    'bedroom': [255, 224, 128],
    'hall': [255, 160, 96],
    'balcony': [255, 224, 224],
    'not used 1': [255, 255, 255],
    'not used 2': [255, 255, 255],
    'door & window': [255, 60, 128],
    'wall': [0, 0, 0]
}


def ind2rgb(ind_im, color_map=floorplan_map):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))
    for i, rgb in color_map.items():
        rgb_im[(ind_im == i)] = rgb
    return rgb_im


def main(args):
    # load input
    im = imageio.imread(args.im_path, pilmode='RGB')
    plt.imshow(im)
    plt.show()
    im = makeImageBetter(im)
    im = im.astype(np.float32) / 255.0  # Normalize to [0, 1]
    im = np.array(Image.fromarray(
        (im * 255).astype(np.uint8)).resize((512, 512))) / 255.0

    # create tensorflow session
    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.Session() as sess:
        # initialize
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        # restore pretrained model
        saver = tf.compat.v1.train.import_meta_graph(
            './pretrained/pretrained_r3d.meta')
        saver.restore(sess, './pretrained/pretrained_r3d')

        # get default graph
        graph = tf.compat.v1.get_default_graph()

        # restore inputs & outputs tensor
        x = graph.get_tensor_by_name('inputs:0')
        room_type_logit = graph.get_tensor_by_name('Cast:0')
        room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

        # infer results
        [room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],
                                              feed_dict={x: im.reshape(1, 512, 512, 3)})
        room_type, room_boundary = np.squeeze(
            room_type), np.squeeze(room_boundary)

        # merge results
        floorplan = room_type.copy()
        floorplan[room_boundary == 1] = 9
        floorplan[room_boundary == 2] = 10
        floorplan_rgb = ind2rgb(floorplan)

        # plot results
        plt.subplot(121)
        plt.imshow(im)
        plt.subplot(122)
        plt.imshow(floorplan_rgb / 255.0)  # Normalize to [0, 1] for display

        legend_handles = [Patch(color=np.array(color) / 255.0, label=label)
                          for label, color in floorplan_legend.items()]
        plt.legend(handles=legend_handles, bbox_to_anchor=(
            1.05, 1), loc='upper left', borderaxespad=0.)

        plt.show()


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
