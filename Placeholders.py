'''
So far we have used Variables to manage our data, but there is a more basic structure, the placeholder. 
A placeholder is simply a variable that we will assign data to at a later date. It allows us to create our
 operations and build our computation graph, without needing the data. In TensorFlow terminology, 
 we then feed data into the graph through these placeholders.
'''
import tensorflow as tf
tf.reset_default_graph() 
x = tf.placeholder("float", None)
y = x * 2

with tf.Session() as session:
    writer = tf.summary.FileWriter("/tmp/basic", session.graph)
    merged = tf.summary.merge_all()
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    result1=session.run(y, feed_dict={x: [10, 20, 30]})

    print(result)