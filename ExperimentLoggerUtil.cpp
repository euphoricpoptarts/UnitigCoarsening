#include <vector>
#include <stdlib.h>
#include <iostream>
#include <fstream>

class ExperimentLoggerUtil {

private:
	class CoarseLevel {
	public:
		int refineIterations = 0;
		bool iterationMaxReached = false;
		uint64_t unrefinedEdgeCut = 0;

		CoarseLevel(int refineIterations, bool iterationMaxReached, uint64_t unrefinedEdgeCut) :
			refineIterations(refineIterations),
			iterationMaxReached(iterationMaxReached),
			unrefinedEdgeCut(unrefinedEdgeCut) {}

		void log(std::ofstream& f) {
			f << "{";
			f << "'refine-iterations':" << refineIterations << ',';
			f << "'iteration-max-reached':" << iterationMaxReached << ',';
			f << "'unrefined-edge-cut':" << unrefinedEdgeCut;
			f << "}";
		}
	};

	int numCoarseLevels = 0;
	std::vector<CoarseLevel> coarseLevels;
	uint64_t finestEdgeCut = 0;
	uint64_t partitionDiff = 0;
	double totalDurationSeconds = 0;
	double coarsenDurationSeconds = 0;
	double refineDurationSeconds = 0;
	double coarsenSortDurationSeconds = 0;

public:
	void addCoarseLevel(int refineIterations, bool iterationMaxReached, uint64_t unrefinedEdgeCut) {
		coarseLevels.emplace_back(refineIterations, iterationMaxReached, unrefinedEdgeCut);
		numCoarseLevels++;
	}

	void setFinestEdgeCut(uint64_t finestEdgeCut) {
		this->finestEdgeCut = finestEdgeCut;
	}

	void setPartitionDiff(uint64_t partitionDiff) {
		this->partitionDiff = partitionDiff;
	}

	void setTotalDurationSeconds(double totalDurationSeconds) {
		this->totalDurationSeconds = totalDurationSeconds;
	}

	void setCoarsenDurationSeconds(double coarsenDurationSeconds) {
		this->coarsenDurationSeconds = coarsenDurationSeconds;
	}

	void setRefineDurationSeconds(double refineDurationSeconds) {
		this->refineDurationSeconds = refineDurationSeconds;
	}

	void setCoarsenSortDurationSeconds(double coarsenSortDurationSeconds) {
		this->coarsenSortDurationSeconds = coarsenSortDurationSeconds;
	}

	void log(char* filename, bool first, bool last) {
		std::ofstream f;
		f.open(filename, std::ios::app);

		if (f.is_open()) {
			if (first) {
				f << "[";
			}
			f << "{";
			f << "'edge-cut':" << finestEdgeCut << ',';
			f << "'partition-diff':" << partitionDiff << ',';
			f << "'total-duration-seconds':" << totalDurationSeconds << ',';
			f << "'coarsen-duration-seconds':" << coarsenDurationSeconds << ',';
			f << "'refine-duration-seconds':" << refineDurationSeconds << ',';
			f << "'coarsen-sort-duration-seconds':" << coarsenSortDurationSeconds << ',';
			f << "'number-coarse-levels':" << numCoarseLevels << ',';
			f << "'coarse-levels:':[";

			bool firstLog = true;
			for (CoarseLevel l : coarseLevels) {
				if (!firstLog) {
					f << ',';
				}
				l.log(f);
				firstLog = true;
			}

			f << "]";
			f << "}";
			if (!last) {
				f << ",";
			}
			else {
				f << "]";
			}
			f.close();
		}
		else {
			std::cerr << "Could not open " << filename << std::endl;
		}
	}
};