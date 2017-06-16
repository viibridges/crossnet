import tensorflow as tf
import misc, config, crossnet

data_dir  = "data/"
data_list = data_dir + "data.csv"

default_config = config.default_config
size_dict      = config.SizeContainer()

def main(_):
  
  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    network = crossnet.CrossNet(sess)
    network.load_data(data_list, data_dir)

    misc.pprint(network.config.__flags)

    network.train_test()

if __name__ == '__main__':
  tf.app.run()
