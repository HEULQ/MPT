#!/usr/local/bin/python

''' 
MPT model used in Yim, H, Dennis, S. J., & Sloutsky, V. M. (2013)
The Development of Episodic Memory: Items, Contexts, and Relations, Psychological Science
@author: Hyugwook Yim (yim.31@osu.edu) 2013/3/16
:see http://cogdev.osu.edu/mpt/code.php for more descriptions

############################################################################
##############################   MAIN   ####################################
# examples of using (1) the parameter estimation function (fitMPT)
#                   (2) the bootstrapping function (bootstrap)
#               and (3) the randomization test function (rand_test)
#
# - All examples use 4yr data in Experiment1.
# - However, other data sets in the paper are available 
#    (i.e. Ex1_7y, Ex2_4y, Ex2_7y, Ex2_a) as well as raw data for bootstrap 
#   and rand_test (e.g. Ex1_4y_ABCD, ...)
############################################################################
'''

from MPT_data import *
from MPT_function import *

#>>> 0> Merge individual data for group parameter estimation.
# - the 'fitMPT' function estimates group data. 
# Therefore, individual data should be aggregated into a list
#
#>>> Experiment 1 (merge the three conditions into one list)
Ex1_4y = list(numpy.sum(Ex1_4y_ABCD,0))+list(numpy.sum(Ex1_4y_ABAC,0))+list(numpy.sum(Ex1_4y_ABABr,0))
Ex1_7y = list(numpy.sum(Ex1_7y_ABCD,0))+list(numpy.sum(Ex1_7y_ABAC,0))+list(numpy.sum(Ex1_7y_ABABr,0))
#>>> Experiment 2 (merge the three conditions into one list)
Ex2_4y = list(numpy.sum(Ex2_4y_ABCD,0))+list(numpy.sum(Ex2_4y_ABAC,0))+list(numpy.sum(Ex2_4y_ABABr,0))
Ex2_7y = list(numpy.sum(Ex2_7y_ABCD,0))+list(numpy.sum(Ex2_7y_ABAC,0))+list(numpy.sum(Ex2_7y_ABABr,0))
Ex2_a = list(numpy.sum(Ex2_ay_ABCD,0))+list(numpy.sum(Ex2_ay_ABAC,0))+list(numpy.sum(Ex2_ay_ABABr,0))


#>>> (1) Estimates parameters 
# "fitMPT(data)"
# > Executes parameter optimization
# > Prints loglikelihood/BIC/AIC, estimated parameters, and the raw data
# > The function takes raw data which is stroed in MPT_data.py
#   - raw data from MPT_data.py should be summed into group data (one list of 13 values)
#
# >>> Example
# - The example uses 4yr data from Experiment1 
#

fitMPT(Ex1_4y)

#>>> 2> Bootstrapping the parameters
# "bootstrap(NumSample, dataABCD, dataABAC, dataABABr, savefile)"
# > Executes bootstraping on the 4 paramters
# > Prints 5 percentile values and saves the bootstrapped samples
# > The function takes 5 arguments
#  - NumSample: Number of samples for bootstrapping
#  - dataABCD, dataABAC, dataABABr: should contain individual raw data
#  - savefile: if you leave the savefile option to 0 it will not save a file
# ## note that bootstrapping will take some time depending on the sample size...
#
# >>> Example
# - The example uses 4yr data from Experiment1 with a sample size of 100
#   (but you might want a larger sample size which would take more computing time)
# - It also saves the results to a file '4yr_MPT_boot.txt'
#   

NumSamples = 100
bootstrap(NumSamples, Ex1_4y_ABCD, Ex1_4y_ABAC, Ex1_4y_ABABr, '4yr_MPT_boot.txt')


#>>> 3> Randomization test 
# "rand_test(NumSample, data1ABCD, data1ABAC, data1ABABr, data2ABCD, data2ABAC, data2ABABr)"
# > Executes a randomization test and compares 4 parameters between two groups
# > The function takes 7 arguments
# - NumSample: Number of samples for bootstrapping
# - data1ABCD, ~ data2ABABr: should contain individual raw data
#   data1 comes from one group and data2 comes from another group
# ## note that the randomization test will take some time depending on the sample size...
#
# >>> Example
# - The example compares 4yr and 7yr data from Experiment1 with a sample size of 100
#   (but you might want a larger sample size for better accuracy 
#                                         which would take more computing time)
#   

NumSamples = 100
rand_test(NumSamples, Ex1_4y_ABCD, Ex1_4y_ABAC, Ex1_4y_ABABr, Ex1_7y_ABCD, Ex1_7y_ABAC, Ex1_7y_ABABr)

