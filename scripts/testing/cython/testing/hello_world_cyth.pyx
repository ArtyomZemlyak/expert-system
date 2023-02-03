
def say_hello_to_cy(str name):
    return "Hello %s!" % name

cdef int my_loop(int i):
    cdef int a = 0
    for j in range(i):
        a += j * j
    return a

def my_2loop(int i):
    cdef int a = 0
    for j in range(i):
        a += my_loop(j)
    return a
