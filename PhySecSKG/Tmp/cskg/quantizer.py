import statistics
from typing import List, Union
import numpy as np


class Quantizer:

    def __init__(self, profile, deviation):
        self.profile = profile
        self.deviation = deviation

    def quantize_stddev(self) -> object:
        """

        :rtype: object
        """

        preliminary_key = []
        # self.profile = (np.array(self.profile, float)).tolist()
        mv = statistics.mean(self.profile)
        # upper = self.deviation*statistics.median_high(self.profile)
        assert isinstance(self.profile, object)
        # lower = self.deviation*statistics.median_low(self.profile)
        upper = mv + self.deviation * statistics.stdev(self.profile)
        lower = mv - self.deviation * statistics.stdev(self.profile)
        for value in self.profile:
            if value > upper:
                preliminary_key.append(1)
                preliminary_key.append(1)
            elif upper > value > mv:
                preliminary_key.append(0)
                preliminary_key.append(1)
            elif value < mv and value < lower:
                preliminary_key.append(1)
                preliminary_key.append(0)
                # prelimnaryKey.append(1)
            else:
                preliminary_key.append(0)
                preliminary_key.append(0)
        return preliminary_key

    def block_quantize_stddev(self) -> object:
        """
        :rtype: object
        """
        val_per_block = 10
        preliminary_key = []
        tmp_profile = list(self.profile)

        #reminder = len(tmp_profile) % val_per_block

        #for index in range(reminder):
        tmp_profile.append(float(0.0))
        tmp_profile.append(float(0.0))

        num_of_blocks = int(len(tmp_profile) / val_per_block)

        for block_index in range(num_of_blocks):
            start_index = val_per_block * block_index
            end_index = val_per_block * (block_index + 1)
            new_list = tmp_profile[start_index:end_index]
            mv = statistics.mean(new_list)
            upper = mv + self.deviation * statistics.stdev(new_list)
            lower = mv - self.deviation * statistics.stdev(new_list)

            for value in new_list:
                if value > upper:
                    preliminary_key.append(1)
                    preliminary_key.append(1)
                elif upper > value > mv:
                    preliminary_key.append(0)
                    preliminary_key.append(1)
                elif value < mv and value < lower:
                    preliminary_key.append(1)
                    preliminary_key.append(0)
                    # prelimnaryKey.append(1)
                else:
                    preliminary_key.append(0)
                    preliminary_key.append(0)
        #print("end")
        return preliminary_key[:-4]


    def quantize_var(self) -> object:
        """

        :rtype: object
        """
        #print(len(self.profile))

        preliminary_key = []
        # self.profile = (np.array(self.profile, float)).tolist()
        mv = statistics.mean(self.profile)
        # upper = self.deviation*statistics.median_high(self.profile)
        assert isinstance(self.profile, object)
        # lower = self.deviation*statistics.median_low(self.profile)
        upper = mv + (statistics.variance(self.profile) / 2)
        lower = mv - (statistics.variance(self.profile) / 2)
        for value in self.profile:
            if value > upper:
                preliminary_key.append(1)
                preliminary_key.append(1)
            elif upper > value > mv:
                preliminary_key.append(0)
                preliminary_key.append(1)
            elif value < mv and value < lower:
                preliminary_key.append(1)
                preliminary_key.append(0)
                # prelimnaryKey.append(1)
            else:
                preliminary_key.append(0)
                preliminary_key.append(0)
        return preliminary_key


    def block_quantize_var(self) -> object:
        """
        :rtype: object
        """
        val_per_block = 10
        preliminary_key = []
        tmp_profile = list(self.profile)

        #reminder = len(tmp_profile) % val_per_block

        #for index in range(reminder):
         #   tmp_profile.append(float(0.0))
        tmp_profile.append(float(0.0))
        tmp_profile.append(float(0.0))

        num_of_blocks = int(len(tmp_profile) / val_per_block)

        for block_index in range(num_of_blocks):
            start_index = val_per_block * block_index
            end_index = val_per_block * (block_index + 1)
            new_list = tmp_profile[start_index:end_index]
            mv = statistics.mean(new_list)
            upper = mv + (statistics.variance(new_list) / 2)
            lower = mv - (statistics.variance(new_list) / 2)

            for value in new_list:
                if value > upper:
                    preliminary_key.append(1)
                    preliminary_key.append(1)
                elif upper > value > mv:
                    preliminary_key.append(0)
                    preliminary_key.append(1)
                elif value < mv and value < lower:
                    preliminary_key.append(1)
                    preliminary_key.append(0)
                    # prelimnaryKey.append(1)
                else:
                    preliminary_key.append(0)
                    preliminary_key.append(0)
        #print("end")
        return preliminary_key[:-4]

    def quantize_median(self):
        preliminary_key = []
        median = statistics.median(self.profile)
        for value in self.profile:
            if value > median:
                preliminary_key.append(1)
            else:
                preliminary_key.append(0)
        return preliminary_key

    def block_quantize_median(self) -> object:
        """

        :rtype: object
        """
        val_per_block = 10
        preliminary_key = []
        tmp_profile = list(self.profile)
        reminder = len(tmp_profile) % val_per_block
        tmp_profile.append(float(0.0))
        tmp_profile.append(float(0.0))
        #for index in range(reminder):
        #    tmp_profile.append(float(0.0))

        num_of_blocks = int(len(tmp_profile) / val_per_block)

        for block_index in range(num_of_blocks):
            start_index = val_per_block * block_index
            end_index = val_per_block * (block_index + 1)
            new_list = tmp_profile[start_index:end_index]
            median = statistics.median(new_list)
            for value in new_list:
                if value > median:
                    preliminary_key.append(1)
                else:
                    preliminary_key.append(0)
        return preliminary_key[:-2]

    def quantize_mean(self):
        preliminary_key = []
        #print(len(self.profile))

        mean = statistics.mean(self.profile)
        for value in self.profile:
            if value > mean:
                preliminary_key.append(1)
            else:
                preliminary_key.append(0)
        return preliminary_key

    def block_quantize_mean(self) -> object:
        """

        :rtype: object
        """
        val_per_block = 10
        preliminary_key = []
        tmp_profile = list(self.profile)

        #reminder = len(tmp_profile) % val_per_block

        #for index in range(reminder):
        #    tmp_profile.append(float(0.0))
        tmp_profile.append(float(0.0))
        tmp_profile.append(float(0.0))
        num_of_blocks = int(len(tmp_profile) / val_per_block)

        for block_index in range(num_of_blocks):
            start_index = val_per_block * block_index
            end_index = val_per_block * (block_index + 1)
            new_list = tmp_profile[start_index:end_index]
            mean = statistics.mean(new_list)
            for value in new_list:
                if value > mean:
                    preliminary_key.append(1)
                else:
                    preliminary_key.append(0)
        return preliminary_key[:-2]