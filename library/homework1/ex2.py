import random

x = 1.0
u = 10 ** (-16)
y = z = u
print('Adunare:')
print('(x + y) + z = {}'.format((x + y) + z))
print('x + (y + z) = {}'.format(x + (y + z)))

c = 0
while True:
    c += 1
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    z = random.uniform(0, 1)
    if x * y * z != x * (y * z):
        print('Inmultire:')
        print('x * (y * z) = {}'.format(x * (y * z)))
        print('x * y * z = {}'.format(x * y * z))
        print('x = {}; y = {}; z = {}'.format(x, y, z))
        break
print('Incercari: {}'.format(c))
