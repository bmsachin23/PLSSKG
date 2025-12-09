import numpy as np


class Bitdisagreement:

    def bdr(self, key1, key2):
        leng = len(key1)
        key = np.bitwise_xor(key2, key1)
        d = 0;
        position = []
        # print("dis - ", key)
        for i in range(leng):
            if key[i] == 1:
                d = d + 1
                position.append(i)
        return d, (d / leng) * 100, position
