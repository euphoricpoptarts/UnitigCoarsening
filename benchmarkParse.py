import sys
import os
from parse import parse
from statistics import mean, stdev

def main():

    filepath = sys.argv[1]

    print("analysing data for {}".format(filepath))

    form = "{} a {} {} {} {}"
    refineIters = []
    totalTimes = []
    coarsenTimes = []
    refineTimes = []
    edgeCuts = []

    cnt = 0
    with open(filepath) as fp:
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