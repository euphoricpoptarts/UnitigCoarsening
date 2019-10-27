#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 csr_filename metrics_filename"
    exit
fi
testFile=$1
metricsFile=$2
for i in {1..100}
do
    ./sgpar $testFile $metricsFile
done

totalT=0
coarsenT=0
refineT=0
line=0
while IFS=',' read -r total coarsen refine
do
    totalT=$(echo $totalT + $total | bc)
    coarsenT=$(echo $coarsenT + $coarsen | bc)
    refineT=$(echo $refineT + $refine | bc)
    line=$((line + 1))
done < "$metricsFile"
averageTotal=$(echo "scale=5; $totalT/$line" | bc)
averageCoarsen=$(echo "scale=5; $coarsenT/$line" | bc)
averageRefine=$(echo "scale=5; $refineT/$line" | bc)
echo "average time = $averageTotal, average coarsen = $averageCoarsen, average refine = $averageRefine, total lines = $line"