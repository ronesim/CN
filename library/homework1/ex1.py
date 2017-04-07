for i in range(10000):
    x = 10 ** (-i)
    if 1.0 + x == 1.0:
        print('u = {}\nm = {}'.format(10 ** (-i), i))
        break
