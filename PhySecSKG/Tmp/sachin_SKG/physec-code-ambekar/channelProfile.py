import csv
from typing import List


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]


class ChannelProfile:
    # To create a rssi, rsrp or pwr profile
    def __init__(self, file):
        self.file = file

    def readCSV(self, rownumber):
        profile: List[float] = []
        lines = [line.rstrip('\n') for line in open(self.file, encoding="utf8", errors='ignore')]
        counter = 0
        for line in lines[1:]:
            try:
                value = float(line.split(",")[rownumber])
                profile.append(value)
                counter += 1
            except ValueError:
                print(line.split(",")[rownumber])
            if 128 == counter:
                break
        return profile
