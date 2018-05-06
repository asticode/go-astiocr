import tensorflow as tf
import json
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('data_directory_path', '', 'Path to the data directory')


def create_tf_record(image):

    # Basics
    height = image["height"]
    width = image["width"]
    filename = image["path"].encode()
    with tf.gfile.GFile(image['path'], 'rb') as fid:
        encoded_image = fid.read()
    image_format = 'png'.encode()

    # Init slices
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # Loop through boxes
    for box in image['boxes']:
        xmins.append(float(box['x0'] / width))
        xmaxs.append(float(box['x1'] / width))
        ymins.append(float(box['y0'] / height))
        ymaxs.append(float(box['y1'] / height))
        classes_text.append(box['label'].encode())
        classes.append(box['label_index'])

    # Generate record
    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))


def write_tf_record(path):
    # Create tfrecord writer
    with tf.python_io.TFRecordWriter(path + "/data.record") as w:
        # Open json
        with open(path + "/summary.json") as f:
            data = json.load(f)

        # Loop through images
        for image in data["images"]:
            tf_record = create_tf_record(image)
            w.write(tf_record.SerializeToString())


def main(_):
    # Write records
    write_tf_record(flags.FLAGS.data_directory_path + "/training")
    write_tf_record(flags.FLAGS.data_directory_path + "/test")


if __name__ == '__main__':
    tf.app.run()
