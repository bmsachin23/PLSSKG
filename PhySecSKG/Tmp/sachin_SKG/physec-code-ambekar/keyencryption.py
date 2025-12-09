import hashlib


class Keyencryption:

    def encryptkey(self, prelimkey: object) -> object:
        str1 = ''.join(str(e) for e in prelimkey)
        key = (hashlib.sha3_512(str1.encode()))
        return key.hexdigest()
