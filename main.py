import time
import numpy
time_bef = time.time()
aggregate = 0
for i in range(1, 50):
    array_a = numpy.random.random((1000, 1000))
    array_b = numpy.random.random((1000, 1000))
    product = array_a * array_b
time_aft = time.time()
time_for = time_aft - time_bef
print(time_for)