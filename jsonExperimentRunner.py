import sys
import os
import subprocess
from glob import glob
from parse import parse
from statistics import mean, stdev, median
import secrets
import json
from pathlib import Path
from threading import Thread, BoundedSemaphore
from itertools import zip_longest

parCall = "./sgpar_exp"
serialCall = "./sgpar_serial"
bigParCall = "./sgpar_hg_exp"
bigSerialCall = "./sgpar_hg_serial"

rateLimit = BoundedSemaphore(value = 4)
waitLimit = 300

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
    coarsenCountTimes = [d['coarsen-count-duration-seconds'] for d in data]
    coarsenPrefixSumTimes = [d['coarsen-prefix-sum-duration-seconds'] for d in data]
    coarsenBucketTimes = [d['coarsen-bucket-duration-seconds'] for d in data]
    coarsenDedupeTimes = [d['coarsen-dedupe-duration-seconds'] for d in data]
    coarsenMapTimes = [d['coarsen-map-duration-seconds'] for d in data]
    coarsenBuildTimes = [d['coarsen-build-duration-seconds'] for d in data]
    refineTimes = [d['refine-duration-seconds'] for d in data]
    edgeCuts = [d['edge-cut'] for d in data]
    edgeCuts4way = [d['edge-cut-four-way'] for d in data]
    coarseLevels = list(zip_longest(*[reversed(d['coarse-levels']) for d in data]))
    numCoarseLevels = [d['number-coarse-levels'] for d in data]
    numCoarseLevels = list(map(lambda x: x - 1, numCoarseLevels))

    with open(logFile, "w") as output:
        printStat("Total duration", times, output)
        printStat("Coarsening duration", coarsenTimes, output)
        printStat("Coarsening mapping duration", coarsenMapTimes, output)
        printStat("Coarsening building duration", coarsenBuildTimes, output)
        printStat("Coarsening counting duration", coarsenCountTimes, output)
        printStat("Coarsening prefix sum duration", coarsenPrefixSumTimes, output)
        printStat("Coarsening bucketing duration", coarsenBucketTimes, output)
        printStat("Coarsening deduping duration", coarsenDedupeTimes, output)
        printStat("Refine duration", refineTimes, output)
        printStat("Edge cut", edgeCuts, output)
        printStat("Four partition edge cut", edgeCuts4way, output)
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

def runExperiment(executable, filepath, metricDir, logFile, t_count):

    myenv = os.environ.copy()
    myenv['OMP_NUM_THREADS'] = str(t_count)

    metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
    call = [executable, filepath, metricsPath, "base_config.txt"]
    call_str = " ".join(call)
    with rateLimit:
        print("running {}".format(call_str), flush=True)
        process = subprocess.Popen(call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=myenv)
        try:
            returncode = process.wait(waitLimit=timeout)
        except TimeoutExpired:
            proc.kill()
            print("{} has timed out!".format(call_str), flush=True)

    if(returncode != 0):
        print("error code: {}".format(returncode))
        print("error produced by:")
        print(call_str, flush=True)
    else:
        analyzeMetrics(metricsPath, logFile)

def processGraph(filepath, metricDir, logFileTemplate):
    
    logFile = logFileTemplate.format("parHEC")
    runExperiment(parCall, filepath, metricDir, logFile, 8)

    #logFile = logFileTemplate.format("serialHEC")
    #runExperiment(serialCall, filepath, metricDir, logFile, t_count)

    print("end {} processing".format(filepath), flush=True)

def reprocessMetricsFromLogFile(f_path):
    form = "running {} sgpar on csr/{}.csr, data logged in {}"
    reprocessList = []
    with open(f_path) as fp:
        for line in fp:
            r = parse(form, line)
            if r != None:
                reprocess = {}
                reprocess['metrics'] = r[2]
                reprocess['log'] = "redo_stats/" + r[1] + "_" + r[0].replace(" ","_") + ".txt"
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

    threads = []
    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        #will fill in the third argument later
        logFile = "{}/{}_{}_Sampling_Data.txt".format(logDir, stem,"{}")
        t = Thread(target=processGraph, args=(filepath, metricDir, logFile))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()