import sys
import os
from parse import parse
from statistics import mean, stdev
import secrets

def main():

    filepath = sys.argv[1]
    metricsPath = "metrics/group{}.txt".format(secrets.token_urlsafe(10))

    sysCall = "./sgpar {} {} 5 0 0 100 > /dev/null"

    print("running sgpar on {}".format(filepath))

    os.system(sysCall.format(filepath, metricsPath))

    form = "{} a {} {} {} {}"
    refineIters = []
    totalTimes = []
    coarsenTimes = []
    refineTimes = []
    edgeCuts = []

    cnt = 0
    with open(metricsPath) as fp:
       for line in fp:
           parsed = parse(form, line)
           refineIters.append(int(parsed[0]))
           totalTimes.append(float(parsed[1]))
           coarsenTimes.append(float(parsed[2]))
           refineTimes.append(float(parsed[3]))
           edgeCuts.append(int(parsed[4]))
           cnt += 1

    print("mean refine iterations: {}".format(mean(refineIters)))
    print("min refine iterations: {}".format(min(refineIters)))
    print("refine iterations std deviation: {}".format(stdev(refineIters)))
    print("mean total time: {}".format(mean(totalTimes)))
    print("min total time: {}".format(min(totalTimes)))
    print("mean edge cut: {}".format(mean(edgeCuts)))
    print("min edge cut: {}".format(min(edgeCuts)))
    print("edge cut std deviation: {}".format(stdev(edgeCuts)))

if __name__ == "__main__":
    main()