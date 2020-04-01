import sys
import os
from glob import glob
from parse import parse
from statistics import mean, stdev, median
import secrets
import json
from pathlib import Path
from threading import Thread

parCall = "./sgpar_hg_exp {} {} 0 0 0 10 > /dev/null"
serialCall = "./sgpar_hg_serial {} {} {} 0 0 10 > /dev/null"
set_threads = "export OMP_NUM_THREADS={}"

def printStat(fieldTitle, statList, outfile):
    min_s = min(statList)
    max_s = max(statList)
    avg = mean(statList)
    sdev = stdev(statList)
    med = median(statList)
    print("{}: mean={}, median={}, min={}, max={}, std-dev={}".format(fieldTitle, avg, med, min_s, max_s, sdev), file=outfile)

def analyzeMetrics(metricsPath, logFile):
    with open(metricsPath) as fp:
            data = json.load(fp)

    times = [d['total-duration-seconds'] for d in data]
    coarsenTimes = [d['coarsen-duration-seconds'] for d in data]
    coarsenSortTimes = [d['coarsen-sort-duration-seconds'] for d in data]
    refineTimes = [d['refine-duration-seconds'] for d in data]
    edgeCuts = [d['edge-cut'] for d in data]
    coarseLevels = list(zip(*[d['coarse-levels'] for d in data]))
    numCoarseLevels = [d['number-coarse-levels'] for d in data]
    coarsenNonSortTimes = [i - j for i, j in zip(coarsenTimes, coarsenSortTimes)]

    with open(logFile, "w") as output:
        printStat("Total duration", times, output)
        printStat("Coarsening duration", coarsenTimes, output)
        printStat("Coarsening sort duration", coarsenSortTimes, output)
        printStat("Coarsening other duration", coarsenNonSortTimes, output)
        printStat("Refine duration", refineTimes, output)
        printStat("Edge cut", edgeCuts, output)
        printStat("Coarse levels", numCoarseLevels, output)
        coarseLevel = len(coarseLevels) - 1
        for level in coarseLevels:
            printStat("Coarse level {} refine iterations".format(coarseLevel), [l['refine-iterations'] for l in level], output)
            printStat("Coarse level {} unrefined edge cuts".format(coarseLevel), [l['unrefined-edge-cut'] for l in level], output)
            coarseLevel = coarseLevel - 1

def processGraph(filepath, metricDir, logFileTemplate):
    
    #parallel HEC
    metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
    print("running parallel HEC sgpar on {}, data logged in {}".format(filepath, metricsPath))
    err = os.system(parCall.format(filepath, metricsPath))
    if(err == 256):
        print("error code: {}".format(err))
        print("error produced by:")
        print(parCall.format(filepath, metricsPath))
    else:
        logFile = logFileTemplate.format("parHEC")
        analyzeMetrics(metricsPath, logFile)

    #serial HEC
    metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
    print("running serial HEC sgpar on {}, data logged in {}".format(filepath, metricsPath))
    err = os.system(serialCall.format(filepath, metricsPath, 0))
    if(err == 256):
        print("error code: {}".format(err))
        print("error produced by:")
        print(serialCall.format(filepath, metricsPath, 0))
    else:
        logFile = logFileTemplate.format("serialHEC")
        analyzeMetrics(metricsPath, logFile)

    #serial matching
    metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
    print("running serial match sgpar on {}, data logged in {}".format(filepath, metricsPath))
    err = os.system(serialCall.format(filepath, metricsPath, 1))
    if(err == 256):
        print("error code: {}".format(err))
        print("error produced by:")
        print(serialCall.format(filepath, metricsPath, 1))
    else:
        logFile = logFileTemplate.format("serialMatch")
        analyzeMetrics(metricsPath, logFile)

    print("end {} processing".format(filepath))

def main():

    dirpath = sys.argv[1]
    metricDir = sys.argv[2]
    logDir = sys.argv[3]
    globMatch = "{}/*.csr".format(dirpath)

    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        #will fill in the third argument later
        logFile = "{}/{}_{}_Sampling_Data.txt".format(logDir, stem,"{}")
        processGraph(filepath, metricDir, logFile)


if __name__ == "__main__":
    main()