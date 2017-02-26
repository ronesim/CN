import random
import time
from math import factorial as f, sin as s, pi

c1 = 1 / f(3)
c2 = 1 / f(5)
c3 = 1 / f(7)
c4 = 1 / f(9)
c5 = 1 / f(11)
c6 = 1 / f(13)

def p1(x):
   y = x * x
   return x * (1 + y * (-c1 + c2 * y))

def p2(x):
   y = x * x
   return x * (1 + y * (-c1 + y * (c2 - c3 * y)))

def p3(x):
   y = x * x
   return x * (1 + y * (-c1 + y * (c2 - y * (-c3 + c4 * y))))

def p4(x):
   y = x * x
   return x * (1 + y * (-0.166 + y * (0.00833 - y * (-c3 + c4 * y))))

def p5(x):
   y = x * x
   return x * (1 + y * (-c1 + y * (c2 - y * (-c3 + y * (c4 - c5 * y)))))

def p6(x):
   y = x * x
   return x * (1 + y * (-c1 + y * (c2 - y * (-c3 + y * (c4 + y * (-c5 + c6 * y))))))

polinoame = [p1, p2, p3, p4, p5, p6]

freq = [0, 0, 0, 0, 0, 0]

MAX = 100000

save_nr = []

for i in range(MAX):
    nr = random.uniform(-pi / 2, pi / 2)
    save_nr.append(nr)
    aux = []
    true_val = s(nr)
    for x in range(6):
        aux.append((x, abs(polinoame[x](nr) - true_val)))
    rem = sorted(aux, key=lambda k: k[1])[:3]
    for item in rem:
        freq[item[0]] += 1

print(freq)

for p in range(6):
    timp = time.time()
    for nr in save_nr:
        polinoame[p](nr)
    print('Polinom {}: {} secunde'.format(p + 1, time.time() - timp))
