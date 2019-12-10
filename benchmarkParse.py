import sys
import os
from glob import glob
from parse import parse
from statistics import mean, stdev
import secrets
from pathlib import Path
from threading import Thread


tolerances = ["1e-1","1e-2","1e-3","1e-4","1e-5","1e-6","1e-7","1e-8","1e-9","1e-10"]
sysCall = "./sgpar {} {} 5 0 0 100 {} {} > /dev/null"
form = "{} a {} {} {} {} {}"
refineForm = "{} {}"

def concatStatWithSpace(stats, stat):
    return ' '.join(map(str, stats[stat]))

def processGraph(filepath, bestPart, logFile):
    statsByTol = {}
    statsByTol["refineMean"] = []
    statsByTol["refineMin"] = []
    statsByTol["refineStdDev"] = []
    statsByTol["totalMaxIter"] = []
    statsByTol["timeMean"] = []
    statsByTol["timeMin"] = []
    statsByTol["edgeCutMean"] = []
    statsByTol["edgeCutMin"] = []
    statsByTol["edgeCutStdDev"] = []
    statsByTol["swapsMean"] = []
    statsByTol["swapsMin"] = []
    statsByTol["swapsStdDev"] = []
    for tolerance in tolerances:
        metricsPath = "metrics/group{}.txt".format(secrets.token_urlsafe(10))
        print("running sgpar on {} with tol {}, comparing to {}, data logged in {}".format(filepath, tolerance, bestPart, metricsPath))

        err = os.system(sysCall.format(filepath, metricsPath, bestPart, tolerance))

        if(err == 256):
            print("error code: {}".format(err))
            print("error produced by:")
            print(sysCall.format(filepath, metricsPath, bestPart, tolerance))
        else:
            refineIters = []
            totalTimes = []
            coarsenTimes = []
            refineTimes = []
            edgeCuts = []
            partSwaps = []
            maxIterReached = 0

            cnt = 0
            with open(metricsPath) as fp:
                for line in fp:
                    parsed = parse(form, line)
                    refineData = parse(refineForm, parsed[0])
                    refineIters.append(int(refineData[0]))
                    maxIterReached += int(refineData[1])
                    totalTimes.append(float(parsed[1]))
                    coarsenTimes.append(float(parsed[2]))
                    refineTimes.append(float(parsed[3]))
                    edgeCuts.append(int(parsed[4]))
                    partSwaps.append(int(parsed[5]))
                    cnt += 1

            statsByTol["refineMean"].append(mean(refineIters))
            statsByTol["refineMin"].append(min(refineIters))
            statsByTol["refineStdDev"].append(stdev(refineIters))
            statsByTol["totalMaxIter"].append(maxIterReached)
            statsByTol["timeMean"].append(mean(totalTimes))
            statsByTol["timeMin"].append(min(totalTimes))
            statsByTol["edgeCutMean"].append(mean(edgeCuts))
            statsByTol["edgeCutMin"].append(min(edgeCuts))
            statsByTol["edgeCutStdDev"].append(stdev(edgeCuts))
            statsByTol["swapsMean"].append(mean(partSwaps))
            statsByTol["swapsMin"].append(min(partSwaps))
            statsByTol["swapsStdDev"].append(stdev(partSwaps))

    output = open(logFile, "w")
    print("tolerances: {}".format(' '.join(tolerances)), file=output)
    print("mean refine iterations: {}".format(concatStatWithSpace(statsByTol, "refineMean")), file=output)
    print("min refine iterations: {}".format(concatStatWithSpace(statsByTol, "refineMin")), file=output)
    print("refine iterations std deviation: {}".format(concatStatWithSpace(statsByTol, "refineStdDev")), file=output)
    print("times max iter reached: {}".format(concatStatWithSpace(statsByTol, "totalMaxIter")), file=output)
    print("mean total time: {}".format(concatStatWithSpace(statsByTol, "timeMean")), file=output)
    print("min total time: {}".format(concatStatWithSpace(statsByTol, "timeMin")), file=output)
    print("mean edge cut: {}".format(concatStatWithSpace(statsByTol, "edgeCutMean")), file=output)
    print("min edge cut: {}".format(concatStatWithSpace(statsByTol, "edgeCutMin")), file=output)
    print("edge cut std deviation: {}".format(concatStatWithSpace(statsByTol, "edgeCutStdDev")), file=output)
    print("mean swaps to best partition: {}".format(concatStatWithSpace(statsByTol, "swapsMean")), file=output)
    print("min swaps to best partition: {}".format(concatStatWithSpace(statsByTol, "swapsMin")), file=output)
    print("swaps to best partition std deviation: {}".format(concatStatWithSpace(statsByTol, "swapsStdDev")), file=output)
    print("end {} processing".format(filepath))

def main():

    dirpath = sys.argv[1]
    bestPartDir = sys.argv[2]
    logDir = sys.argv[3]
    globMatch = "{}/*.csr".format(dirpath)
    threads = []

    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        bestPart = "{}/{}.2.ptn".format(bestPartDir, stem)
        logFile = "{}/{}_Sampling_Data.txt".format(logDir, stem)
        #processGraph(filepath, bestPart, logFile)
        t = Thread(target=processGraph, args=(filepath, bestPart, logFile,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()