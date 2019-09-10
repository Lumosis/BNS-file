#!/bin/bash
path=$(pwd)

if [ $1 == 0 ]; then
	rm $path/simulator/logs/*.*
	rm $path/simulator/outputs/*.*
	rm $path/*.png
	rm $path/TM.txt
	rm $path/Topology.txt
	rm $path/Results*.txt
	rm $path/Flow_Results*.txt
	rm $path/Pickled_Results*.txt
	rm $path/OptimalSolution.txt
	rm $path/*.pyc
	rm $path/simulator/data/job*.tsv
	rm $path/Results/TRAIN/*.txt*
	rm $path/Results/TEST/*.txt*
	rm $path/*.dat
	rm $path/timeline_01.json
	rm -r $path/gen-trace
	rm -r $path/Results/CHECKPOINTS
	rm -r $path/Results/TF_STATS/train_*
	rm -r $path/RotorNetSolution
	rm -r $path/mprofile_*
	rm -r $path/nr
	rm -r $path/plots
	rm -r $path/xWeaver
	rm -r $path/SolsticeAlgorithm
	rm -r $path/HeuristicSolution
	rm -r $path/EdmondsAlgorithm
	rm -r $path/OptimalSolution
	rm -r $path/RANDOM
	rm -r $path/sim-eval-*
	rm -r $path/bazel-*
	rm -r $path/learningModels/*.pyc
	rm -r $path/simulator/*.pyc
	rm -r $path/RotorNet*
fi

if [ $1 == 1 ]; then
	python2 RunMe.py
fi