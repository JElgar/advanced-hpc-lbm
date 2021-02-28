# Makefile

EXE=d2q9-bgk

CC=icc
# CFLAGS= -std=c99 -Wall -O3
# omp 
CFLAGS = -std=c99 -Wall -Ofast -mtune=native -fopenmp -no-prec-sqrt 
# Debug
# CFLAGS = -std=c99 -Wall -Ofast -mtune=native -fopenmp -g
# CFLAGS = -std=c99 -Wall -Ofast -mtune=native -fopenmp -g -qopt-report=5 -qopt-report-phase=vec
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

profile-run: $(EXE).c
	$(CC) $(CFLAGS) -pg $^ $(LIBS) -o $(EXE)

profile-generate: $(EXE).c
	gprof $(EXE) gmon.out > profile.txt

submit-clean:
	make clean
	make
	sbatch job_submit_d2q9-bgk

sc:
	make submit-clean


check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
