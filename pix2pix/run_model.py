import argparse
import cv2
import numpy as np
import tensorflow as tf



def load_graph(frozen_graph_filename):
    graph=tf.Graph()
    with graph.as_default():
        od_graph_def=tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename,'rb') as fid:
            serialized_graph=fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def,name='')
    return  graph

def main():
    graph=load_graph("frozen_model.pb")
    input_image=graph.get_tensor_by_name('input_image:0')
    output_image=graph.get_tensor_by_name('generator/output_image:0')
    sess=tf.Session(graph=graph)
    src=cv2.imread("922.png")
    src=cv2.resize(src,(256,256), interpolation=cv2.INTER_LINEAR)
    generated_image=sess.run(output_image,feed_dict={input_image:src})
    image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
   #cv2.imshow("test",image)
    print(image_bgr.shape)
    cv2.imshow('test',image_bgr)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()