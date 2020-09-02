import sys
import os
import re
from glob import glob
from parse import parse
from statistics import mean, stdev, median
import secrets
import json
from pathlib import Path
from itertools import zip_longest

stemParse = r"(.+)_fm_(.+)_(.+)_Sampling_Data"
lineParse = "{}mean={}, median={}, min={}, max={}, std-dev={}"
#fieldHeaders = "Graph, HEC, MIS, Match, MT"
fieldHeaders = "Graph, method, 1, 2, 4, 8, 16, 32"

def textStatsToCSV(stem, filepath, relevantLines):
    graphStats = []
    parsedLines = []
    with open(filepath,"r") as f:
        for line in f:
            parsed = parse(lineParse,line)
            parsedLines.append(parsed)

    if len(parsedLines) < 11:
        return stem

    desiredStats = [2,5]
    for line in relevantLines:
        parsed = parsedLines[line]

        chosenStats = []
        if parsed is not None:
            parsed = list(parsed)
            for stat in desiredStats:
                chosenStats.append(parsed[stat])
        else:
            for stat in desiredStats:
                chosenStats.append("nan")
        graphStats.append(chosenStats)
    return graphStats

def main():

    logDir = sys.argv[1]
    outFile = sys.argv[2]

    globMatch = "{}/*.txt".format(logDir)

    data = {}
    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        stemMatch = re.match(stemParse, stem)
        if stemMatch is not None:
            graph = (stemMatch.groups()[0],stemMatch.groups()[1])
            experiment = stemMatch.groups()[2]
            if graph not in data:
                data[graph] = {}
            #if experiment in ["HEC", "MIS"]:
            #    data[graph][experiment] = textStatsToCSV(stem, filepath, [0,1,2,3,8,9])
            #else:
            #0 is total, 1 is all coarsen, 2 is coarsen map, 3 is coarsen build, 12 is refine time, 13 is cutsize
            data[graph][experiment] = textStatsToCSV(stem, filepath, [0,1,2,3,12,13,15])

    experiments = ["1","2","4","8","16","32"]#["hec","mis","match","mt"]
    with open(outFile,"w") as f:
        print(fieldHeaders, file=f)
        for graph, values in data.items():
            l = [graph[0], graph[1]]
            for experiment in experiments:
                if experiment in values:
                    l.append("{}".format(values[experiment][1][0]))
                else:
                    l.append("DNF")
            #for experimentName, experiment in values.items():
            #    print(",".join([graph, experimentName[0], experimentName[1], experiment[2][0]]), file=f)
            print(",".join(l), file=f)


if __name__ == "__main__":
    main()
