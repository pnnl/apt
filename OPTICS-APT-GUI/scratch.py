# a place for scratchy test

import numpy as np
import pickle

class TestClass(object):
    def __init__(self):
        self.x = 1
        self.y = np.array((1,2,3))

    def mutate(self):
        rand_idx = np.random.randint(0, 3)
        self.y[rand_idx] = 100

    def __str__(self):
        str_x = str(self.x)
        str_y = np.array2string(self.y)
        return 'str_x is '+str_x+'\n'+'str_y is '+str_y


Tester = TestClass()
print(Tester)

out_fname = 'Tester_class.obj'

with open(out_fname, 'wb') as f:
    pickle.dump(Tester, f)


Tester_2 = pickle.load(open(out_fname, 'rb'))
print(Tester_2)

Tester_2.mutate()
print(Tester_2)

Tester_3 = pickle.load(open(out_fname, 'rb'))
print(Tester_3)