import tensorflow as tf


# first approach is simple matrices using the tf.constant
# a,b,c are all 1x3 matrices, d is the calculated function
a = tf.constant([1, 2, 3, 1, 2, 3])
b = tf.constant([3, 2, 1, 1, 2, 3])
c = tf.constant([4, 5, 6, 1, 2, 3])
d = (a*a + b) * c

sess = tf.Session()  # open session

print("simple 1x3 matrix", sess.run(d), '\n')

#  declaring variables
x = tf.fill([5, 5], 5.6)  # setting up matrix x,y,z as fill
y = tf.fill([5, 5], 10.4)
z = tf.fill([5, 5], 75.33)
r = tf.constant([2.12345], shape=[5, 5])   # different type of matrix multiplication to show using fill and constant
pow_x = tf.pow(x, 2)                  # using the power function
add_xx_y = tf.add(pow_x, y)           # using the add function
mul_z = tf.multiply(add_xx_y, z)      # using the multiply function to scale the
mul_r = tf.multiply(add_xx_y, r)

# using with loop to run the session, print the matrices, and the results from the matrix algebra form above
with tf.Session() as sess:
    print('matrix x', sess.run(x), '\n')
    print('matrix y', sess.run(y), '\n')
    print('matrix z', sess.run(z), '\n')
    print('matrix r', sess.run(r), '\n')
    print('matrix z as scalar\n')
    print(sess.run(mul_z), '\n')
    print('matrix r as scalar\n')
    print(sess.run(mul_r))

sess.close()  # Close the tensorflow session






