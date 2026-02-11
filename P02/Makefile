SHELL := bash
CC ?= gcc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra
OMPFLAG ?= -fopenmp
PY ?= python3

.PHONY: all clean run plot

all: p02

p02: p02.c
	$(CC) $(CFLAGS) $(OMPFLAG) -o $@ $<

# Bash-driver sweep (like in class), N = 2^0 .. 2^18
# Parallel runs use OMP_NUM_THREADS as the *max*; program prints powers-of-two <= that.
run: p02
	@export OMP_PLACES=threads; \
	export OMP_PROC_BIND=true; \
	export OMP_NUM_THREADS=128; \
	out=p02_out.txt; \
	: > $$out; \
	i=0; \
	while [ $$i -le 18 ]; do \
		N=$$((2**$$i)); \
		echo "Processing $$N or 2^$$i" | tee -a $$out; \
		./p02 $$N | tee -a $$out; \
		echo "" | tee -a $$out; \
		i=$$(($$i + 1)); \
	done

plot: run plot_p02.py
	$(PY) plot_p02.py p02_out.txt p02_plot.png p02_plot.csv
	@echo "Wrote p02_plot.png and p02_plot.csv"

clean:
	rm -f p02 p02_out.txt p02_plot.png p02_plot.csv
