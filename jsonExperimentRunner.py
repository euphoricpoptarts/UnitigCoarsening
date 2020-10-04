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

cudaCall = "./sgpar.cuda"
hecCall = "./sgpar.hec"
mtCall = "./sgpar.mt"
mpCall = "./sgpar.mp"
matchCall = "./sgpar.match"
misCall = "./sgpar.mis"
serialCall = "./sgpar_serial"
bigParCall = "./sgpar_hg_exp"
bigSerialCall = "./sgpar_hg_serial"

rateLimit = BoundedSemaphore(value = 1)
waitLimit = 10800

def printStat(fieldTitle, statList, outfile):
    min_s = min(statList)
    max_s = max(statList)
    avg = mean(statList)
    sdev = "only one data-point"
    if len(statList) > 1:
        sdev = stdev(statList)
    med = median(statList)
    print("{}: mean={}, median={}, min={}, max={}, std-dev={}".format(fieldTitle, avg, med, min_s, max_s, sdev), file=outfile)

def listToStats(statList):
    stats = {}
    stats["min"] = min(statList)
    stats["max"] = max(statList)
    stats["mean"] = mean(statList)
    stats["std-dev"] = "only one data-point"
    if len(statList) > 1:
        stats["std-dev"] = stdev(statList)
    stats["median"] = median(statList)
    return stats

def dictToStats(data):
    output = {}
    for key, value in data.items():
        if isinstance(value[0], dict):
            d = []
            for datum in value:
                d.append(dictToStats(datum))
            output[key] = d
        else:
            output[key] = listToStats(value)
    return output


def printDict(data, outfile):
    for key, value in data.items():
        if isinstance(value[0], dict):
            for idx, datum in enumerate(value):
                print("{} Level {}:".format(key, idx), file=outfile)
                printDict(datum, outfile)
        else:
            printStat(key, value, outfile)

def transposeListOfDicts(data):
    data = [x for x in data if x is not None]
    transposed = {}
    if len(data) == 0:
        return transposed
    #all entries should have same fields
    fields = [x for x in data[0]]
    for field in fields:
        transposed[field] = [datum[field] for datum in data]

    fieldsToTranpose = []
    for key, value in transposed.items():
        if len(value) > 0 and isinstance(value[0], list):
            fieldsToTranpose.append(key)

    for field in fieldsToTranpose:
        #value is a list of lists, transform it into list of dicts
        aligned_lists = zip_longest(*transposed[field])
        dict_list = []
        for l in aligned_lists:
            d = transposeListOfDicts(l)
            dict_list.append(d)
        transposed[field] = dict_list
    return transposed

def analyzeMetrics(metricsPath, logFile):
    with open(metricsPath) as fp:
        data = json.load(fp)

    data = transposeListOfDicts(data)

    with open(logFile, "w") as output:
        printDict(data, output)

    statsDict = dictToStats(data)
    jsonFile = os.path.splitext(logFile)[0] + ".json"
    with open(jsonFile, "w") as output:
        json.dump(statsDict, output)

def analyzeMetricsOld(metricsPath, logFile):
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
    coarsenPermuteTimes = [d['coarsen-permute-duration-seconds'] for d in data]
    coarsenMapConstructTimes = [d['coarsen-map-construct-duration-seconds'] for d in data]
    coarsenRadixSortTimes = [d['coarsen-radix-sort-duration-seconds'] for d in data]
    coarsenRadixDedupeTimes = [d['coarsen-radix-dedupe-duration-seconds'] for d in data]
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
        printStat("Coarsening permuting duration", coarsenPermuteTimes, output)
        printStat("Coarsening map constructing duration", coarsenMapConstructTimes, output)
        printStat("Coarsening radix sorting duration", coarsenRadixSortTimes, output)
        printStat("Coarsening radix deduping duration", coarsenRadixDedupeTimes, output)
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

    if(os.path.exists(logFile)):
        return
    myenv = os.environ.copy()
    myenv['OMP_NUM_THREADS'] = str(t_count)

    metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
    call = [executable, filepath, metricsPath, "base_config.txt"]
    call_str = " ".join(call)
    with rateLimit:
        print("running {}".format(call_str), flush=True)
        stdout_f = "/var/tmp/sgpar_log.txt"
        with open(stdout_f, 'w') as fp:
            process = subprocess.Popen(call, stdout=fp, stderr=subprocess.DEVNULL, env=myenv)
        try:
            returncode = process.wait(timeout = waitLimit)
        except subprocess.TimeoutExpired:
            process.kill()
            print("Timeout reached by {}".format(call_str), flush=True)
            return

    if(returncode != 0):
        print("error code: {}".format(returncode))
        print("error produced by:")
        print(call_str, flush=True)
    else:
        analyzeMetrics(metricsPath, logFile)

def processGraph(filepath, metricDir, logFileTemplate):
    
    #for t in [1,2,4,8,16,32]:
    logFile = logFileTemplate.format("spec_hec")#.format(t))
    runExperiment(cudaCall, filepath, metricDir, logFile, 64)
        #logFile = logFileTemplate.format("fm_mt_{}".format(t))
        #runExperiment(mtCall, filepath, metricDir, logFile, t)
        #logFile = logFileTemplate.format("fm_match_{}".format(t))
        #runExperiment(matchCall, filepath, metricDir, logFile, t)
        #logFile = logFileTemplate.format("fm_mis_{}".format(t))
        #runExperiment(misCall, filepath, metricDir, logFile, t)

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
