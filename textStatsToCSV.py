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
import math

stemParse = r"(.+)_(fm|spec)_(.*)_Sampling_Data"
lineParse = "{}mean={}, median={}, min={}, max={}, std-dev={}"
#fieldHeaders = "Graph, HEC, MIS, Match, MT"
fieldHeaders = "{Graph} & {hec} & {match} & {mtmetis} & {gosh} & {mis2}"

def getStats(filepath):
    with open(filepath,"r") as f:
       return json.load(f) 

def gpuBuildTable(graphs,data,outFile):
    with open(outFile,"w+") as f:
        print(fieldHeaders, file=f)
        for graph in graphs:
            graphSanitized = graph.replace("_","\\textunderscore ")
            l = [graphSanitized]
            values = data[graph]
            exp = values[("spec","hec")]
            exp_gemm = values[("spec","hec_gemm")]
            exp_map = values[("spec","hec_hashmap")]
            l.append("{:.2f}".format(exp["coarsen-duration-seconds"]["median"]))
            l.append("{:.2f}".format(exp["coarsen-build-duration-seconds"]["median"] / exp["coarsen-duration-seconds"]["median"]))
            l.append("{:.2f}".format(exp_map["coarsen-build-duration-seconds"]["median"] / exp["coarsen-build-duration-seconds"]["median"]))
            l.append("{:.2f}".format(exp_gemm["coarsen-build-duration-seconds"]["median"] / exp["coarsen-build-duration-seconds"]["median"]))
            print(" & ".join(l) + " \\\\", file=f)

def cpuBuildTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print(fieldHeaders, file=f)
        for graph in graphs:
            graphSanitized = graph.replace("_","\\textunderscore ")
            l = [graphSanitized]
            values = data[graph]
            exp = values[("spec","hecCPU")]
            l.append("{:.2f}".format(exp["coarsen-duration-seconds"]["median"]))
            l.append("{:.2f}".format(exp["coarsen-build-duration-seconds"]["median"] / exp["coarsen-duration-seconds"]["median"]))
            try:
                exp_map = values[("spec","hecCPU_hashmap")]
                l.append("{:.2f}".format(exp_map["coarsen-build-duration-seconds"]["median"] / exp["coarsen-build-duration-seconds"]["median"]))
            except KeyError:
                l.append("DNF")
            try:
                exp_gemm = values[("spec","hecCPU_gemm")]
                l.append("{:.2f}".format(exp_gemm["coarsen-build-duration-seconds"]["median"] / exp["coarsen-build-duration-seconds"]["median"]))
            except KeyError:
                l.append("DNF")
            print(" & ".join(l) + " \\\\", file=f)

def gpuvcpuTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print(fieldHeaders, file=f)
        for graph in graphs:
            graphSanitized = graph.replace("_","\\textunderscore ")
            l = [graphSanitized]
            values = data[graph]
            gpu = values[("spec","hec")]
            cpu = values[("spec","hecCPU")]
            l.append("{:.2f}".format(cpu["coarsen-duration-seconds"]["median"] / gpu["coarsen-duration-seconds"]["median"]))
            l.append("{:.2f}".format(cpu["coarsen-build-duration-seconds"]["median"] / gpu["coarsen-build-duration-seconds"]["median"]))
            try:
                serial = values[("spec","hecCPUSerial")]
                l.append("{:.2f}".format(serial["coarsen-duration-seconds"]["median"] / cpu["coarsen-duration-seconds"]["median"]))
                l.append("{:.2f}".format(serial["coarsen-build-duration-seconds"]["median"] / cpu["coarsen-build-duration-seconds"]["median"]))
            except KeyError:
                l.append("DNF")
                l.append("DNF")
            print(" & ".join(l) + " \\\\", file=f)

def methodsCompTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print(fieldHeaders, file=f)
        for graph in graphs:
            graphSanitized = graph.replace("_","\\textunderscore ")
            l = [graphSanitized]
            values = data[graph]
            main = values[("spec","hec")]
            otherExps = ["match","mtmetis","gosh","mis2"]
            for other in otherExps:
                try:
                    exp = values[("spec",other)]
                    l.append("{:.2f}".format(exp["coarsen-duration-seconds"]["median"] / main["coarsen-duration-seconds"]["median"]))
                except KeyError:
                    l.append("DNF")
            
            l.append("{:.2f}".format(main["number-coarse-levels"]["median"]))
            for other in otherExps:
                try:
                    exp = values[("spec",other)]
                    l.append("{:.0f}".format(exp["number-coarse-levels"]["median"]))
                except KeyError:
                    l.append("DNF")
            
            hecCoarseLevels = main["coarse-levels"]
            hecCoarsenRate = hecCoarseLevels[-1]["number-vertices"]["median"] / hecCoarseLevels[0]["number-vertices"]["median"]
            hecCoarsenRate = math.pow(hecCoarsenRate, 1/(main["number-coarse-levels"]["median"] - 1))
            l.append("{:.2f}".format(hecCoarsenRate))
           
            try:
                mtmetis = values[("spec","mtmetis")]
                mtCL = mtmetis["coarse-levels"]
                mtCR = mtCL[-1]["number-vertices"]["median"] / mtCL[0]["number-vertices"]["median"]
                mtCR = math.pow(mtCR, 1/(mtmetis["number-coarse-levels"]["median"] - 1))
                l.append("{:.2f}".format(mtCR))
            except KeyError:
                l.append("DNF")

            print(" & ".join(l) + " \\\\", file=f)

def main():

    logDir = sys.argv[1]
    outFile = sys.argv[2]

    globMatch = "{}/*.json".format(logDir)

    data = {}
    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        stemMatch = re.match(stemParse, stem)
        if stemMatch is not None:
            graph = stemMatch.groups()[0]
            experiment = (stemMatch.groups()[1], stemMatch.groups()[2])
            if graph not in data:
                data[graph] = {}
            data[graph][experiment] = getStats(filepath)

    graphsSorted = [key for key in data]
    graphsSorted = sorted(graphsSorted, key = str.casefold)
    gpuBuildTable(graphsSorted, data, outFile)
    cpuBuildTable(graphsSorted, data, outFile)
    gpuvcpuTable(graphsSorted, data, outFile)
    methodsCompTable(graphsSorted, data, outFile)

if __name__ == "__main__":
    main()
