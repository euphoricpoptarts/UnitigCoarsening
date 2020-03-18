import sys
import os
from glob import glob
from parse import parse
from statistics import mean, stdev
import secrets
import json
from pathlib import Path
from threading import Thread

sysCall = "./sgpar_hg_exp {} {} 0 0 0 10 > /dev/null"

def printStat(fieldTitle, statList, outfile):
    min_s = min(statList)
    max_s = max(statList)
    avg = mean(statList)
    sdev = stdev(statList)
    print("{}: mean={}, min={}, max={}, std-dev={}".format(fieldTitle, min_s, max_s, avg, sdev), file=outfile)

def processGraph(filepath, metricDir, logFile):
    
    metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
    print("running sgpar on {}, data logged in {}".format(filepath, metricsPath))

    err = os.system(sysCall.format(filepath, metricsPath))

    if(err == 256):
        print("error code: {}".format(err))
        print("error produced by:")
        print(sysCall.format(filepath, metricsPath, bestPart, tolerance))
    else:

        cnt = 0
        with open(metricsPath) as fp:
            data = json.load(fp)

        times = [d['total-duration-seconds'] for d in data]
        coarsenTimes = [d['coarsen-duration-seconds'] for d in data]
        coarsenSortTimes = [d['coarsen-sort-duration-seconds'] for d in data]
        refineTimes = [d['refine-duration-seconds'] for d in data]
        edgeCuts = [d['edge-cut'] for d in data]
        coarseLevels = list(zip(*[d['coarse-levels'] for d in data]))
        numCoarseLevels = [d['number-coarse-levels'] for d in data]

        with open(logFile, "w") as output:
            printStat("Total duration", times, output)
            printStat("Coarsening duration", coarsenTimes, output)
            printStat("Coarsening sort duration", coarsenSortTimes, output)
            printStat("Refine duration", refineTimes, output)
            printStat("Edge cut", edgeCuts, output)
            printStat("Coarse levels", coarseLevels, output)
            coarseLevel = numCoarseLevels - 1
            for level in coarseLevels:
                printStat("Coarse level {} refine iterations".format(coarseLevel), [l['refine-iterations'] for l in level], output)
                coarseLevel = coarseLevel - 1

    print("end {} processing".format(filepath))

def main():

    dirpath = sys.argv[1]
    metricDir = sys.argv[2]
    logDir = sys.argv[3]
    globMatch = "{}/*.csr".format(dirpath)

    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        logFile = "{}/{}_Sampling_Data.txt".format(logDir, stem)
        processGraph(filepath, metricDir, logFile)


if __name__ == "__main__":
    main()