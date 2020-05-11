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
graph experiment, mean time, median time, min time, max time, time std-dev, \
mean coarsen time, median coarsen time, min coarsen time, max coarsen time, coarsen time std-dev, \
mean coarsen sort time, median coarsen sort time, min coarsen sort time, max coarsen sort time, coarsen sort time std-dev, \
mean coarsen other time, median coarsen other time, min coarsen other time, max coarsen other time, coarsen other time std-dev, \
mean refine time, median refine time, min refine time, max refine time, refine time std-dev, \
mean edge cut, median edge cut, min edge cut, max edge cut, edge cut std-dev"

def textStatsToCSV(stem, filepath):
    graphStats = [stem]
    with open(filepath,"r") as f:
        lcount = 0
        for line in f:
            lcount += 1
            if lcount > 6:
                break
            parsed = parse(lineParse,line)
            if parsed is not None:
                parsed = list(parsed)
                for stat in parsed[1:6]:
                    graphStats.append(stat)
            else:
                for stat in range(1,6):
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