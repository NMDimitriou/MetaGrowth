# Make file for the parameter estimation of the FK/KSC/KSMD/KSCMD models

VPATH:=parent:.
IDIR1:=parent/include
IDIR2:=include
ODIR1:=parent/obj
ODIR2:=obj
_DEPS1=declarations_parent.cuh
_DEPS2=declarations.cuh
DEPS1:=$(patsubst %,$(IDIR1)/%,$(_DEPS1))
DEPS2:=$(patsubst %,$(IDIR2)/%,$(_DEPS2))
_OBJ1=main_parent.o host_functions_parent.o
_OBJ2=main.o readData.o setDevice.o simulate.o host_functions.o device_functions.o print_message.o VTKsave.o 

PARENT:=parent/main_parent
MAIN:=main
all: $(PARENT) $(MAIN)

CC=g++
NVCC=nvcc
RM=rm -rf

# model: FK/KSC/KSMD/KSCMD/TEST, 
# IC: EXP/LIN/ID/NA, 
# study: CALIBRATION/SA/NONE
# for NRMSE calculation please choose CALIBRATION flag (loads the experimental data and residual functions)
SIMU_OPTION += -DSTUDY=CALIBRATION
SIMU_OPTION += -DMODEL=KSC
#SIMU_OPTION += -DSAVE_DATA     # for speed, comment SAVE_DATA option when calibrating
#SIMU_OPTION += -DNRMSE 			# for speed, comment NRMSE option when calibrating  
SIMU_OPTION += -DIC=EXP

CFLAGS1=-I$(IDIR1) -O3
CFLAGS2=-I$(IDIR2) -O3
CXXFLAGS=-std=c++11 $(SIMU_OPTION)
CUFLAGS=--ptxas-options=-v -arch=sm_70

LDIR=$(CUDA_LIB_PATH)
LIBS=-lcuda -lcudart -L lib64


ifeq "$(findstring STUDY=CALIBRATION, $(SIMU_OPTION))" "STUDY=CALIBRATION"
_OBJ2+=calcResiduals.o status.o calc_ExpSd.o save_nrmse.o
endif
ifeq "$(findstring STUDY=SA, $(SIMU_OPTION))" "STUDY=SA"
_OBJ2+=calcResponse.o
endif

OBJ1:=$(patsubst %,$(ODIR1)/%,$(_OBJ1))
OBJ2:=$(patsubst %,$(ODIR2)/%,$(_OBJ2))

$(ODIR1)/%.o: %.cu $(DEPS1)
	$(NVCC) -c -o $@ $< $(CFLAGS1) $(CXXFLAGS) $(CUFLAGS)

$(PARENT):$(OBJ1)
	$(CC) -o $@ $^ $(CFLAGS1) $(CXXFLAGS) $(LIBS) -Wall -Wextra -Werror

$(ODIR2)/%.o: %.cu $(DEPS2)
	$(NVCC) -c -o $@ $< $(CFLAGS2) $(CXXFLAGS) $(CUFLAGS)

$(MAIN):$(OBJ2)
	$(CC) -o $@ $^ $(CFLAGS2) $(CXXFLAGS) $(LIBS) -Wall -Wextra -Werror



clean:
	$(RM) $(ODIR1)/*.o *~ core*  
	$(RM) $(ODIR2)/*.o *~ core*  

clear:
	$(RM) inIC*.txt inExp*.txt loglike* gpu* ../devID*.txt status.txt *.raw *.out dakota* devID* runs *slice*

