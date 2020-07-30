import sys
import os
from glob import glob
from parse import parse
from statistics import mean, stdev, median
import secrets
import json
from pathlib import Path
from itertools import zip_longest

lineParse = "{}mean={}, median={}, min={}, max={}, std-dev={}"
fieldHeaders = "\
graph experiment, median time, \
median coarsen time, \
median coarsen map time, \
median coarsen sort time, \
median refine time, \
median edge cut, "

fieldHeaders2 = "\
graph experiment, mean time, median time, min time, max time, time std-dev, \
mean coarsen time, median coarsen time, min coarsen time, max coarsen time, coarsen time std-dev, \
mean coarsen sort time, median coarsen sort time, min coarsen sort time, max coarsen sort time, coarsen sort time std-dev, \
mean coarsen other time, median coarsen other time, min coarsen other time, max coarsen other time, coarsen other time std-dev, \
mean refine time, median refine time, min refine time, max refine time, refine time std-dev, \
mean edge cut, median edge cut, min edge cut, max edge cut, edge cut std-dev"

def textStatsToCSV(stem, filepath):
    graphStats = [stem]
    parsedLines = []
    with open(filepath,"r") as f:
        for line in f:
            parsed = parse(lineParse,line)
            parsedLines.append(parsed)

    if len(parsedLines) < 11:
        return stem

    for line in [0,1,2,3,8,9]:
        parsed = parsedLines[line]
        if parsed is not None:
            parsed = list(parsed)
            for stat in [2]:
                graphStats.append(parsed[stat])
        else:
            for stat in [2]:
                graphStats.append("nan")
    return ",".join(graphStats)

def main():

    logDir = sys.argv[1]
    outFile = sys.argv[2]

    globMatch = "{}/*.txt".format(logDir)

    data = []
    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        data.append(textStatsToCSV(stem, filepath))

    with open(outFile,"w") as f:
        print(fieldHeaders, file=f)
        for datum in data:
            print(datum, file=f)


if __name__ == "__main__":
    main()