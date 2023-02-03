import time

# from hello_world_py import say_hello_to_cy as say_hello_to, my_loop, my_2loop
# from hello_world_cy import say_hello_to_cy, my_2loop as my_2loop_cy


def say_hello_to_py(name):
    return "Hello %s!" % name


def my_loop_py(i):
    a = 0
    for j in range(i):
        a += j * j
    return a


def my_2loop_py(i):
    a = 0
    for j in range(i):
        a += my_loop_py(j)
    return a


str_name = "Artem"


st = time.time()
for i in range(1000000):
    a = say_hello_to_py(str_name)
print("Time say_hello_to_py: ", (time.time() - st))  # 0.121         / pypy: 0.01263


# st = time.time()
# for i in range(1000000):
#     a = say_hello_to(str_name)
# print("Time say_hello_to: ", (time.time() - st))        # 0.095


# st = time.time()
# for i in range(1000000):
#     a = say_hello_to_cy(str_name)
# print("Time say_hello_to_cy: ", (time.time() - st))     # 0.089


st = time.time()
a = my_loop_py(1000000)
print("Time my_loop_py: ", (time.time() - st))  # 0.0397          / pypy: 0.001299


st = time.time()
a = my_2loop_py(1000)
print("Time my_2loop_py: ", (time.time() - st))  # 0.0165         / pypy: 0.00159


# st = time.time()
# a = my_loop(1000000)
# print("Time my_loop: ", (time.time() - st))        # 0.02941


# st = time.time()
# a = my_2loop(1000)
# print("Time my_2loop: ", (time.time() - st))        # 0.01097


# st = time.time()
# a = my_2loop_cy(1000)
# print("Time my_2loop_cy: ", (time.time() - st))        # 0.01320
