#!/usr/bin/python

'''
MPT model used in Yim, H, Dennis, S. J., & Sloutsky, V. M. (2013)
The Development of Episodic Memory: Items, Contexts, and Relations, Psychological Science
@author: Hyugwook Yim (yim.31@osu.edu) 2013/3/16
:see http://cogdev.osu.edu/mpt/code for more descriptions

############################################################################
#############################   FUNCTIONs  #################################
############################################################################
'''

import numpy
import math
import random
import scipy.optimize
from scipy import stats

#>>> 1> Computes response patterns when parameters are given
#  The order of the response pattern follows the description on the web-page
#  http://cogdev.osu.edu/mpt/code
############################################################################
def compt_para(e, i, l, b, data):
    totalABCD = sum(data[0:4])
    totalABAC = sum(data[4:9])
    totalABABr = sum(data[9:13])
    g2 = 1.0/2
    g5 = 1.0/5
    g6 = 1.0/6
    # (1) Response patterns from ABCD
    B1 = (e*i*l*b)+(e*i*l*(1-b))+(e*i*(1-l)*b)+(e*i*(1-l)*(1-b))+(e*(1-i)*l*b)+(e*(1-i)*l*(1-b)*g6)+(e*(1-i)*(1-l)*b)+(e*(1-i)*(1-l)*(1-b)*g2*g6)
    DFHJL1 = (e*(1-i)*l*(1-b)*(1-g6))+(e*(1-i)*(1-l)*(1-b)*g2*(1-g6))
    NPRTVX1 = e*(1-i)*(1-l)*(1-b)*(1-g2)
    miss1 = (1-e)
    # (2) Response patterns from ABAC
    B2 = (e*i*l*b)+(e*i*l*(1-b))+(e*i*(1-l)*b)+(e*i*(1-l)*(1-b)*g2)+(e*(1-i)*l*b)+(e*(1-i)*l*(1-b)*g6)+(e*(1-i)*(1-l)*b)+(e*(1-i)*(1-l)*(1-b)*g2*g6)
    DFHJL2 = (e*(1-i)*l*(1-b)*(1-g6))+(e*(1-i)*(1-l)*(1-b)*g2*(1-g6))
    N2 = (e*i*(1-l)*(1-b)*(1-g2))+(e*(1-i)*(1-l)*(1-b)*(1-g2)*g6)
    PRTVX2 = e*(1-i)*(1-l)*(1-b)*(1-g2)*(1-g6)
    miss2 = (1-e)
    # (3) Response patterns from ABABr
    B3 = (e*i*l*b)+(e*i*l*(1-b)*g2)+(e*i*(1-l)*b)+(e*i*(1-l)*(1-b)*g2)+(e*(1-i)*l*b)+(e*(1-i)*l*(1-b)*g6)+(e*(1-i)*(1-l)*b)+(e*(1-i)*(1-l)*(1-b)*g6)
    D3 = (e*i*l*(1-b)*(1-g2))+(e*i*(1-l)*(1-b)*(1-g2))+(e*(1-i)*l*(1-b)*(1-g6)*g5)+(e*(1-i)*(1-l)*(1-b)*(1-g6)*g5)
    FHJL3 = (e*(1-i)*l*(1-b)*(1-g6)*(1-g5))+(e*(1-i)*(1-l)*(1-b)*(1-g6)*(1-g5))
    miss3 = (1-e)
    results = [B1*totalABCD, DFHJL1*totalABCD, NPRTVX1*totalABCD, miss1*totalABCD,
               B2*totalABAC, DFHJL2*totalABAC, N2*totalABAC, PRTVX2*totalABAC, miss2*totalABAC,
               B3*totalABABr, D3*totalABABr, FHJL3*totalABABr, miss3*totalABABr]
    print "(5) Compare results"
    print ">> the order of response patterns are described at http://cogdev.osu.edu/mpt/code"
    print "Model: ",
    for i in xrange(len(results)):
        print ("%0.1f    " % results[i]),
    print ""
    print "Data : ",
    for i in xrange(len(data)):
        print ("%0.1f    " % data[i]),
    print " "
    return(results)


#####################################
#>>> 2> Computes minus log-likelihood
def mpt_mle(x, data):
    # To make sure the optimizing algorithm doesn't crash
    vsn = 1e-7 #need a Very Small Number so that math.log doesn't get zero values
    for i in range(4):
        if x[i] <= 0.0:
            x[i] = vsn
        if x[i] >= 1.0:
            x[i] = 1.0-vsn
    g2 = 1.0/2
    g5 = 1.0/5
    g6 = 1.0/6
    e = x[0]
    i = x[1]
    l = x[2]
    b = x[3]

    # calculate the proportion of responses for each condition
    # this is due to the fact that we are optimizing the 3 conditions together
    nSum = numpy.sum(data) * 1.0
    nABCD = numpy.sum(data[0:4]) /nSum
    nABAC = numpy.sum(data[4:9]) /nSum
    nABABr = numpy.sum(data[9:13]) /nSum

    # log-likelihood calculation for each condition
    ABCD = (data[0]*math.log((e*i*l*b)+(e*i*l*(1-b))+(e*i*(1-l)*b)+(e*i*(1-l)*(1-b))+(e*(1-i)*l*b)+(e*(1-i)*l*(1-b)*g6)+(e*(1-i)*(1-l)*b)+(e*(1-i)*(1-l)*(1-b)*g2*g6))
          + data[1]*math.log((e*(1-i)*l*(1-b)*(1-g6))+(e*(1-i)*(1-l)*(1-b)*g2*(1-g6)))
          + data[2]*math.log(e*(1-i)*(1-l)*(1-b)*(1-g2))
          + data[3]*math.log((1-e))
           )
    ABAC = (data[4]*math.log((e*i*l*b)+(e*i*l*(1-b))+(e*i*(1-l)*b)+(e*i*(1-l)*(1-b)*g2)+(e*(1-i)*l*b)+(e*(1-i)*l*(1-b)*g6)+(e*(1-i)*(1-l)*b)+(e*(1-i)*(1-l)*(1-b)*g2*g6))
          + data[5]*math.log((e*(1-i)*l*(1-b)*(1-g6))+(e*(1-i)*(1-l)*(1-b)*g2*(1-g6)))
          + data[6]*math.log((e*i*(1-l)*(1-b)*(1-g2))+(e*(1-i)*(1-l)*(1-b)*(1-g2)*g6))
          + data[7]*math.log(e*(1-i)*(1-l)*(1-b)*(1-g2)*(1-g6))
          + data[8]*math.log((1-e))
           )
    ABABr= (data[9]*math.log((e*i*l*b)+(e*i*l*(1-b)*g2)+(e*i*(1-l)*b)+(e*i*(1-l)*(1-b)*g2)+(e*(1-i)*l*b)+(e*(1-i)*l*(1-b)*g6)+(e*(1-i)*(1-l)*b)+(e*(1-i)*(1-l)*(1-b)*g6))
          + data[10]*math.log((e*i*l*(1-b)*(1-g2))+(e*i*(1-l)*(1-b)*(1-g2))+(e*(1-i)*l*(1-b)*(1-g6)*g5)+(e*(1-i)*(1-l)*(1-b)*(1-g6)*g5))
          + data[11]*math.log((e*(1-i)*l*(1-b)*(1-g6)*(1-g5))+(e*(1-i)*(1-l)*(1-b)*(1-g6)*(1-g5)))
          + data[12]*math.log((1-e))
           )
    constant1 = (math.log(math.factorial(nSum) / ( math.factorial(data[0])
                                                 * math.factorial(data[1])
                                                 * math.factorial(data[2])
                                                 * math.factorial(data[3])
                                                 * math.factorial(data[4])
                                                 * math.factorial(data[5]) 
                                                 * math.factorial(data[6]) 
                                                 * math.factorial(data[7]) 
                                                 * math.factorial(data[8]) 
                                                 * math.factorial(data[9]) 
                                                 * math.factorial(data[10])
                                                 * math.factorial(data[11])
                                                 * math.factorial(data[12])
                                                 )#denominator end
                         )#log end
                 )
    constant2 = ( (data[0] + data[1] + data[2] + data[3])*math.log(nABCD)
                + (data[4] + data[5] + data[6] + data[7] + data[8])*math.log(nABAC)
                + (data[9] + data[10] + data[11] + data[12])*math.log(nABABr)
               )
    # THE log-likelihood
    MLE = ABCD + ABAC + ABABr + constant1 + constant2
    # Return it with a minus since we will use an optimizing algorithm that minimizes the function
    return(-MLE)

#####################################################################
#>>> 3> BIC, AIC - get the minus log-lik and turn it into a BIC, AIC
def BICs(MLE, k, n):
    BIC = 2*MLE + k*math.log(n)
    return(BIC)
def AICs(MLE, k, n):
    AIC = 2*MLE + 2*k
    return(AIC)


#######################################
#>>> 4> Functions for fitting the model
# uses scipy.optimize.fmin_l_bfgs_b and prints out MLE, BIC, AIC, & the parameters
# takes data from the global variable 'data'
# after estimating the parameters, compare it with the data given
def fitMPT(data):
    x0 = numpy.random.random(4) 
    xopt = scipy.optimize.fmin_l_bfgs_b(mpt_mle, x0, fprime=None, args = (data,), approx_grad=1, bounds = [(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)])
    print "======================  Optimization  =========================="
    print "(1) loglike: %f" % -xopt[1]
    print "(2)   BIC  : %f" % BICs(xopt[1], 4, numpy.sum(data))
    print "(3)   AIC  : %f" % AICs(xopt[1], 4, numpy.sum(data))
    print "(4) parameter estimation (e, i, l, b)"
    print xopt[0]
    compt_para(xopt[0][0], xopt[0][1], xopt[0][2], xopt[0][3], data)
    
################################################
#>>> 5> Functions for bootstrapping
# Resample the data with replacement then sum it
def reSample(matrix):
    n = matrix.shape[0]
    tempSum = numpy.zeros(matrix.shape[1])
    numPool = range(n) 
    for i in range(n): 
        random.shuffle(numPool)
        tempSum += matrix[numPool[0]]
    return(tempSum)
# Merges the resampled data from each condition (i.e. ABCD, ABAC, ABABr)
def reSample_merged(data1, data2, data3):
    rsampled1 = reSample(data1)
    rsampled2 = reSample(data2)
    rsampled3 = reSample(data3)
    return(list(rsampled1)+list(rsampled2)+list(rsampled3))
# Execute bootstrapping with X iterations
def bootstrap(NumSample, dataABCD, dataABAC, dataABABr, savefile):
    e = []
    i = []
    l = []
    b = []
    #start bootstrapping!
    # the results could be saved in the file below if savefile option is used

    for j in xrange(NumSample):
        x0 = numpy.random.random(4) 
        data = reSample_merged(dataABCD, dataABAC, dataABABr)
        xopt = scipy.optimize.fmin_l_bfgs_b(mpt_mle, x0, fprime=None, args = (data,), approx_grad=1, bounds = [(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)])
        e.append(xopt[0][0])
        i.append(xopt[0][1])
        l.append(xopt[0][2])
        b.append(xopt[0][3])

    if savefile != 0:
        f = open(savefile,'w')
        f.write("e\ti\tl\tb\n")
        for j in xrange(NumSample):
            f.write("%f\t%f\t%f\t%f\n" % (e[j], i[j], l[j], b[j]))    
        f.close()
    
    #(2) or the statistics of the distribution could be calculated
    print "======================  Bootstrapping  ========================="
    print "                          Percentile             "
    print "parameter |    1st      25th      50th      75th      99th "
    print "    e     |  %2.5f   %2.5f   %2.5f   %2.5f   %2.5f" % (stats.scoreatpercentile(e,1),
                                                                stats.scoreatpercentile(e,25),
                                                                stats.scoreatpercentile(e,50),
                                                                stats.scoreatpercentile(e,75), 
                                                                stats.scoreatpercentile(e,99))
    print "    i     |  %2.5f   %2.5f   %2.5f   %2.5f   %2.5f" % (stats.scoreatpercentile(i,1),
                                                                stats.scoreatpercentile(i,25),
                                                                stats.scoreatpercentile(i,50),
                                                                stats.scoreatpercentile(i,75), 
                                                                stats.scoreatpercentile(i,99))
    print "    l     |  %2.5f   %2.5f   %2.5f   %2.5f   %2.5f" % (stats.scoreatpercentile(l,1),
                                                                stats.scoreatpercentile(l,25),
                                                                stats.scoreatpercentile(l,50),
                                                                stats.scoreatpercentile(l,75), 
                                                                stats.scoreatpercentile(l,99))
    print "    b     |  %2.5f   %2.5f   %2.5f   %2.5f   %2.5f" % (stats.scoreatpercentile(b,1),
                                                                stats.scoreatpercentile(b,25),
                                                                stats.scoreatpercentile(b,50),
                                                                stats.scoreatpercentile(b,75), 
                                                                stats.scoreatpercentile(b,99))

##############################################
#>>> 6> Functions for the randomization test
# Mixup the two groups
def mix_vecsum(matA, matB):
    nA = matA.shape[0]
    nB = matB.shape[0]
    matSUM = numpy.concatenate((matA, matB))
    numpy.random.shuffle(matSUM)
    newMatA = sum(matSUM[0:nA])
    newMatB = sum(matSUM[nA:])
    return(newMatA, newMatB)
# Merge the two mixedup groups
def merge_twoRandset(data1ABCD, data1ABAC, data1ABABr, data2ABCD, data2ABAC, data2ABABr):
    abcdA, abcdB = mix_vecsum(data1ABCD,data2ABCD)
    abacA, abacB = mix_vecsum(data1ABAC,data2ABAC)
    ababrA, ababrB = mix_vecsum(data1ABABr,data2ABABr)
    return(list(abcdA)+list(abacA)+list(ababrA), list(abcdB)+list(abacB)+list(ababrB))
# Execute the randomization test
def rand_test(NumSample, data1ABCD, data1ABAC, data1ABABr, data2ABCD, data2ABAC, data2ABABr):
    #calculate the estimated parameters for the two groups
    x0 = numpy.random.random(4) 
    data1 = (list(sum(data1ABCD, 0))+list(sum(data1ABAC,0))+list(sum(data1ABABr,0)))
    xopt1 = scipy.optimize.fmin_l_bfgs_b(mpt_mle, x0, fprime=None, args = (data1,), approx_grad=1, bounds = [(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)])
    x0 = numpy.random.random(4) 
    data2 = (list(sum(data2ABCD, 0))+list(sum(data2ABAC,0))+list(sum(data2ABABr,0)))
    xopt2 = scipy.optimize.fmin_l_bfgs_b(mpt_mle, x0, fprime=None, args = (data2,), approx_grad=1, bounds = [(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)])
    #calculate the parameter difference between the two groups
    para_diff = [xopt1[0][0]-xopt2[0][0], xopt1[0][1]-xopt2[0][1], xopt1[0][2]-xopt2[0][2], xopt1[0][3]-xopt2[0][3]]

    # set storage for the samples
    perm_e = []
    perm_i = []
    perm_l = []
    perm_b = []
    for i in xrange(NumSample):
        #print i
        # randomize the set
        dataA, dataB = merge_twoRandset(data1ABCD, data1ABAC, data1ABABr, data2ABCD, data2ABAC, data2ABABr)
        # optimize for the 1st group
        x0 = numpy.random.random(4) 
        xoptA = scipy.optimize.fmin_l_bfgs_b(mpt_mle, x0, fprime=None, args=(dataA,), approx_grad=1, bounds = [(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)])
        l_A = list(xoptA[0])
        # optimize for the 2nd group
        x0 = numpy.random.random(4) 
        xoptB = scipy.optimize.fmin_l_bfgs_b(mpt_mle, x0, fprime=None, args=(dataB,), approx_grad=1, bounds = [(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)])
        l_B = list(xoptB[0])
        # save them
        perm_e.append(l_A[0] - l_B[0])
        perm_i.append(l_A[1] - l_B[1])
        perm_l.append(l_A[2] - l_B[2])
        perm_b.append(l_A[3] - l_B[3])

    # calculate p-values
    pvalues = []
    pvalues.append(stats.percentileofscore(perm_e, para_diff[0])/100.0)
    pvalues.append(stats.percentileofscore(perm_i, para_diff[1])/100.0)
    pvalues.append(stats.percentileofscore(perm_l, para_diff[2])/100.0)
    pvalues.append(stats.percentileofscore(perm_b, para_diff[3])/100.0)
    for i in xrange(4):
        if pvalues[i] > .5:
            pvalues[i] = 1 - pvalues[i]

    print "============= Randomization test results ======================="
    print "              1st_group minus 2nd_group"
    print " parameter           difference            p-value"
    print "    e     |          %f              %1.3f" % (para_diff[0], pvalues[0])
    print "    i     |          %f              %1.3f" % (para_diff[1], pvalues[1])
    print "    l     |          %f              %1.3f" % (para_diff[2], pvalues[2])
    print "    b     |          %f              %1.3f" % (para_diff[3], pvalues[3])
    print "================================================================"
