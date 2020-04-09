import sys
import os
from glob import glob
from parse import parse
from statistics import mean, stdev, median
import secrets
import json
from pathlib import Path
from threading import Thread
from itertools import zip_longest

parCall = "./sgpar_hg_exp {} {} 0 0 0 10 > /dev/null"
serialCall = "./sgpar_hg_serial {} {} {} 0 0 10 > /dev/null"
set_threads = "export OMP_NUM_THREADS={}"

def printStat(fieldTitle, statList, outfile):
    min_s = min(statList)
    max_s = max(statList)
    avg = mean(statList)
    sdev = "only one data-point"
    if len(statList) > 1:
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
    coarseLevels = list(zip_longest(*[reversed(d['coarse-levels']) for d in data]))
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
        
        for levelIdx in range(0,len(coarseLevels)):
            level = coarseLevels[levelIdx]
            level = list(filter(None, level))
            numVertices = [l['number-vertices'] for l in level]
            if levelIdx > 0:
                fineLevelVertices = [l['number-vertices'] for l in coarseLevels[levelIdx-1]]
                ratios = [ i / j for i, j in zip(numVertices, fineLevelVertices)]
                printStat("Coarse level {} coarsening ratio".format(levelIdx), ratios, output)
            printStat("Coarse level {} number of vertices".format(levelIdx), numVertices, output)
            printStat("Coarse level {} refine iterations".format(levelIdx), [l['refine-iterations'] for l in level], output)
            printStat("Coarse level {} unrefined edge cuts".format(levelIdx), [l['unrefined-edge-cut'] for l in level], output)

def processGraph(filepath, metricDir, logFileTemplate):
    
    #set omp thread count
    os.system(set_threads.format(8))

    #parallel HEC
    metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
    print("running parallel HEC sgpar on {}, data logged in {}".format(filepath, metricsPath), flush=True)
    err = os.system(parCall.format(filepath, metricsPath))
    if(err != 0):
        print("error code: {}".format(err))
        print("error produced by:")
        print(parCall.format(filepath, metricsPath), flush=True)
    else:
        logFile = logFileTemplate.format("parHEC")
        analyzeMetrics(metricsPath, logFile)

    #serial HEC
    metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
    print("running serial HEC sgpar on {}, data logged in {}".format(filepath, metricsPath), flush=True)
    err = os.system(serialCall.format(filepath, metricsPath, 0))
    if(err != 0):
        print("error code: {}".format(err))
        print("error produced by:")
        print(serialCall.format(filepath, metricsPath, 0), flush=True)
    else:
        logFile = logFileTemplate.format("serialHEC")
        analyzeMetrics(metricsPath, logFile)

    #serial matching
    metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
    print("running serial match sgpar on {}, data logged in {}".format(filepath, metricsPath), flush=True)
    err = os.system(serialCall.format(filepath, metricsPath, 1))
    if(err != 0):
        print("error code: {}".format(err))
        print("error produced by:")
        print(serialCall.format(filepath, metricsPath, 1), flush=True)
    else:
        logFile = logFileTemplate.format("serialMatch")
        analyzeMetrics(metricsPath, logFile)

    print("end {} processing".format(filepath), flush=True)

def convert(f_path):
    form = "running {} sgpar on csr/{}.csr, data logged in {}"
    reprocessList = []
    with open(f_path) as fp:
        for line in fp:
            r = parse(form, line)
            if r != None:
                reprocess = {}
                reprocess['metrics'] = r[2]
                reprocess['log'] = "redo_stats/" + r[0].replace(" ","_") + "_" + r[1] + ".txt"
                reprocessList.append(reprocess)

    for reprocess in reprocessList:
        print(reprocess)
        try:
            analyzeMetrics(reprocess['metrics'], reprocess['log'])
        except:
            print("Couldn't process last")

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