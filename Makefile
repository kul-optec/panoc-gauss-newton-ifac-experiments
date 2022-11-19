.PHONY: all clean

all: gn-chain-mpc-times.pdf gn-chain-compare-N-horiz-confidence-parallel.pdf gn-chain-conv-iter.pdf

gn-chain-mpc-times.pdf: gn-chain-mpc-fig.py gn-chain-mpc-GN-cold-avg.pkl gn-chain-mpc-LBFGS-cold-avg.pkl gn-chain-mpc-GN-warm-avg.pkl gn-chain-mpc-LBFGS-warm-avg.pkl
	python $<
gn-chain-mpc-GN-cold-avg.pkl: gn-chain-mpc-parallel.py
	python $< gn cold ||:
gn-chain-mpc-LBFGS-cold-avg.pkl: gn-chain-mpc-parallel.py
	python $< lbfgs cold ||:
gn-chain-mpc-GN-warm-avg.pkl: gn-chain-mpc-parallel.py
	python $< gn warm ||:
gn-chain-mpc-LBFGS-warm-avg.pkl: gn-chain-mpc-parallel.py
	python $< lbfgs warm ||:

gn-chain-compare-N-horiz-confidence-parallel.pdf: gn-chain-compare-N-horiz-confidence-parallel-fig.py gn-chain-compare-N-horiz-confidence-parallel.pkl
	python $<
gn-chain-compare-N-horiz-confidence-parallel.pkl: gn-chain-compare-N-horiz-confidence-parallel.py
	python $< ||:

gn-chain-conv-iter.pdf: gn-chain-conv-iter.py
	python $<

clean:
	rm -f gn-chain-mpc-times.pdf \
		gn-chain-compare-N-horiz-confidence-parallel.pdf \
		gn-chain-conv-iter.pdf \
		gn-chain-mpc-GN-cold-avg.pkl \
		gn-chain-mpc-LBFGS-cold-avg.pkl \
		gn-chain-mpc-GN-warm-avg.pkl \
		gn-chain-mpc-LBFGS-warm-avg.pkl \
		gn-chain-compare-N-horiz-confidence-parallel.pkl
