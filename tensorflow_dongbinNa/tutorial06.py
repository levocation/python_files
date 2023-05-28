import tensorflow as tf
import numpy as np

a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a, b)

print(c) #           tensor 자체를 출력 ===========> tf.Tensor(3, shape=(), dtype=int32)
print(np.int32(c)) # tensor 내부의 value를 출력 ===> 3

print("============================")

v_a = tf.Variable(1)
print(v_a)
v_a = tf.Variable(3)
print(v_a)

print("=====================")

input = np.array(tf.zeros(5))
input[0] += 1
print(input)