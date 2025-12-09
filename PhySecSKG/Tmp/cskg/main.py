import os

from texttable import Texttable
from channelProfile import *
from quantizer import *
from keyencryption import *
from reciprocityEnhancer import *
from bitdisagreement import *

#test
def generatekey(row_number, fileName):
    """

    :param fileName:
    :type row_number: object
    """
    profile_values: List = []

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    fileNameEnb = os.path.join(fileDir, fileName)

    # channelProfile object creation
    profile: ChannelProfile = ChannelProfile(fileNameEnb)
    profile_list = profile.readCSV(row_number)

    # remove unwanted values from the channel profile
    profile_list = remove_values_from_list(profile_list, 0.0)
    profile_list = remove_values_from_list(profile_list, 'nan')

    # enhancing reciprocity
    enhanced_profile = Reciprocityenhancer()

    profile_list = (np.asarray(enhanced_profile.cdt(profile_list)).tolist())[0]

    # quantizer object creation
    prelimKey: Quantizer = Quantizer(profile_list, .5)

    # quantize the channel profile
    stddev_quantise_PK = prelimKey.quantize_stddev()
    var_quantise_PK = prelimKey.quantize_var()
    median_quantize_PK = prelimKey.quantize_median()
    mean_quantize_PK = prelimKey.quantize_mean()

    prelimKey.deviation = 1.25
    block_stddev_quantise_PK = prelimKey.block_quantize_stddev()
    block_var_quantise_PK = prelimKey.block_quantize_var()
    block_median_quantize_PK = prelimKey.block_quantize_median()
    block_mean_quantize_pk = prelimKey.block_quantize_mean()

    # privacy amplification of primary key
    encrypt: Keyencryption = Keyencryption()
    var_quantise_encrypted = encrypt.encryptkey(var_quantise_PK)
    block_var_quantise_encrypted = encrypt.encryptkey(block_var_quantise_PK)
    stddev_quantise_ecnrypted = encrypt.encryptkey(stddev_quantise_PK)
    block_stddev_quantise_encrypted = encrypt.encryptkey(block_stddev_quantise_PK)
    block_median_quantise_encrypted = encrypt.encryptkey(block_median_quantize_PK)
    median_quantize_encrypted = encrypt.encryptkey(median_quantize_PK)
    mean_quantize_encrypted = encrypt.encryptkey(mean_quantize_PK)
    block_mean_quantize_encrypted = encrypt.encryptkey(block_mean_quantize_pk)
    return var_quantise_PK, block_var_quantise_PK, stddev_quantise_PK, block_stddev_quantise_PK, median_quantize_PK, \
           block_median_quantize_PK, mean_quantize_PK, block_mean_quantize_pk



####################################################################################
enb_var_quantise_PK, enb_block_var_quantise_PK, enb_stddev_quantise_PK, enb_block_stddev_quantise_PK, \
enb_median_quantise_PK, enb_block_median_quantise_PK, enb_mean_quantise_PK, enb_block_mean_quantise_PK \
    = generatekey(5, 'mobile_combined.csv')

ue_var_quantise_PK, ue_block_var_quantise_PK, ue_stddev_quantise_PK, ue_block_stddev_quantise_PK, \
ue_median_quantise_PK, ue_block_median_quantise_PK, ue_mean_quantise_PK, ue_block_mean_quantise_PK \
    = generatekey(6, 'mobile_combined.csv')

print("re", enb_block_mean_quantise_PK)

t = Texttable()

check_bdr = Bitdisagreement()
var_quantise_bits, var_quantise_bdr, position_var_blk = check_bdr.bdr(enb_var_quantise_PK, ue_var_quantise_PK)

block_var_quantise_bits, block_var_quantise_bdr, position_var_blk = check_bdr.bdr(enb_block_var_quantise_PK, ue_block_var_quantise_PK)

stddev_quantize_bits, stddev_quantize_bdr, position_var_blk = check_bdr.bdr(enb_stddev_quantise_PK, ue_stddev_quantise_PK)

block_stddev_quantize_bits, block_stddev_quantize_bdr, position_var_blk = check_bdr.bdr(enb_block_stddev_quantise_PK,
                                                                      ue_block_stddev_quantise_PK)

median_quantise_bits, median_quantise_bdr, position_var_blk = check_bdr.bdr(enb_median_quantise_PK, ue_median_quantise_PK)

block_median_quantise_bits, block_median_quantise_bdr, position_var_blk = check_bdr.bdr(enb_block_median_quantise_PK,
                                                                      ue_block_median_quantise_PK)

mean_quantise_bits, mean_quantise_bdr, position_var_blk = check_bdr.bdr(enb_mean_quantise_PK, ue_mean_quantise_PK)

block_mean_quantise_bits, block_mean_quantise_bdr, position_var_blk = check_bdr.bdr(enb_block_mean_quantise_PK,
                                                                  ue_block_mean_quantise_PK)

print("\n")
print("         SKG bit dis-agreement rate of CDT SKG methods")
t.add_rows([['quantisation methods', 'no of disagreed bits', 'Bit-disagreement rate'],
            ['var_quantise_bdr', var_quantise_bits, var_quantise_bdr],
            ['block_var_quantise_bdr', block_var_quantise_bits, block_var_quantise_bdr],
            ['stddev_quantize_bdr', stddev_quantize_bits, stddev_quantize_bdr],
            ['block_stddev_quantize_bdr', block_stddev_quantize_bits, block_stddev_quantize_bdr],
            ['median_quantise_bdr', median_quantise_bits, median_quantise_bdr],
            ['block_median_quantise_bdr', block_median_quantise_bits, block_median_quantise_bdr],
            ['mean_quantise_bdr', mean_quantise_bits, mean_quantise_bdr],
            ['block_mean_quantise_bdr', block_mean_quantise_bits, block_mean_quantise_bdr]])

print(t.draw())

##########################################################################################
print("************************AI*******************")
enhanced_profile = Reciprocityenhancer()
enb, ue = enhanced_profile.avgprofile('trainingdata1.csv', 'mobile_combined.csv', 5, 6)

enb_t = []
for value in enb:
    enb_t.append(value[0])

ue_t = []
for val in ue:
    ue_t.append(val[0])

table2 = Texttable()

enb_t = (np.asarray(enhanced_profile.cdt(enb_t)).tolist())[0]
ue_t = (np.asarray(enhanced_profile.cdt(ue_t)).tolist())[0]

enb_pkey = Quantizer(enb_t, .5)
ue_pkey = Quantizer(ue_t, .5)

enb_var_quantise_PK = enb_pkey.quantize_var()
ue_var_quantise_PK = ue_pkey.quantize_var()

enb_stddev_quantise_PK = enb_pkey.quantize_stddev()
ue_stddev_quantise_PK =ue_pkey.quantize_stddev()

enb_median_quantize_PK = enb_pkey.quantize_median()
ue_median_quantize_PK = ue_pkey.quantize_median()

enb_mean_quantise_PK = enb_pkey.quantize_mean()
ue_mean_quantize_PK = ue_pkey.quantize_mean()

enb_pkey.deviation = 1.5
ue_pkey.deviation = 1.5

enb_block_var_quantise_PK = enb_pkey.block_quantize_var()
ue_block_var_quantise_PK = ue_pkey.block_quantize_var()

enb_block_stddev_quantise_PK = enb_pkey.block_quantize_stddev()
ue_block_stddev_quantise_PK = ue_pkey.block_quantize_stddev()

enb_block_median_quantise_PK = enb_pkey.block_quantize_median()
ue_block_median_quantise_PK = ue_pkey.block_quantize_median()

enb_block_mean_quantise_PK = enb_pkey.block_quantize_mean()
ue_block_mean_quantise_PK = ue_pkey.block_quantize_mean()

check_bdr = Bitdisagreement()

var_quantise_bits, var_quantise_bdr, position_var_blk = check_bdr.bdr(enb_var_quantise_PK, ue_var_quantise_PK)

block_var_quantise_bits, block_var_quantise_bdr, position_var_blk = check_bdr.bdr(enb_block_var_quantise_PK, ue_block_var_quantise_PK)

stddev_quantize_bits, stddev_quantize_bdr, position_std = check_bdr.bdr(enb_stddev_quantise_PK, ue_stddev_quantise_PK)

block_stddev_quantize_bits, block_stddev_quantize_bdr, position_std_blk = check_bdr.bdr(enb_block_stddev_quantise_PK,
                                                                      ue_block_stddev_quantise_PK)

median_quantise_bits, median_quantise_bdr, position_median = check_bdr.bdr(enb_median_quantize_PK, ue_median_quantize_PK)

block_median_quantise_bits, block_median_quantise_bdr, position_median_blk = check_bdr.bdr(enb_block_median_quantise_PK,
                                                                      ue_block_median_quantise_PK)

mean_quantise_bits, mean_quantise_bdr, position_mean = check_bdr.bdr(enb_mean_quantise_PK, ue_mean_quantize_PK)

block_mean_quantise_bits, block_mean_quantise_bdr, position_mean_blk = check_bdr.bdr(enb_block_mean_quantise_PK,
                                                                  ue_block_mean_quantise_PK)
print(enb_block_mean_quantise_PK)

print("\n")
print("         SKG bit dis-agreement rate of AI+DCT SKG methods")
table2.add_rows([['quantisation methods', 'no of disagreed bits', 'Bit-disagreement'],
            ['block_var_quantise_bdr', var_quantise_bits, var_quantise_bdr],
            ['block_quantise_bdr', block_var_quantise_bits, block_var_quantise_bdr],
            ['stddev_quantize_bdr', stddev_quantize_bits, stddev_quantize_bdr],
            ['block_stddev_quantize_bdr', block_stddev_quantize_bits, block_stddev_quantize_bdr],
            ['median_quantise_bdr', median_quantise_bits, median_quantise_bdr],
            ['block_median_quantise_bdr', block_median_quantise_bits, block_median_quantise_bdr],
            ['mean_quantise_bdr', mean_quantise_bits, mean_quantise_bdr],
            ['block_mean_quantise_bdr', block_mean_quantise_bits, block_mean_quantise_bdr]])

print(table2.draw())

##############################################################################################