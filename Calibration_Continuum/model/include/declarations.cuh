#ifndef DECLARATIONS_H
#define DECLARATIONS_H

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <numeric>
#include<sys/shm.h>
#include<sys/stat.h>

using namespace std;

#define NONE	 	0

#define FK       	1
#define KSC      	2
#define KSMD     	3
#define KSCMD    	4
#define TEST		5

#define CALIBRATION 	1
#define SA	    	2

#define EXP		1
#define LIN 	2
#define ID		3
#define NA		4


//structure to hold size of simulation and arrays on the host
typedef struct DataArray
{
	int xmax;
	int ymax;
	int zmax;

#if (MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)	
	double* a; //cell density
	double* a_intm; //intermediate values of a
	double* a_first; //initial density of a (for Alee effect)
	//Parameters for tridiagonal solver
	double* a_rows;
    double* a_cols;
	double* a_ax;
	double* a_ay;
	double* a_az;
	double* a_bx;
	double* a_by;
	double* a_bz;
	double* a_cx;
	double* a_cy;
	double* a_cz;
	//Save slices - not need for tmcmc
    //double* a_slice;
#endif
#if (MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
	double* b; //signal density
	double* b_intm;
	//Parameters for tridiagonal solver
        double* b_rows;
        double* b_cols;
	double* b_ax;
        double* b_ay;
        double* b_az;
        double* b_bx;
        double* b_by;
        double* b_bz;
        double* b_cx;
        double* b_cy;
        double* b_cz;
	double* del_b;  //signal density gradient
	//Save slices - not need for tmcmc
        //double* b_slice;
#endif
#if ( MODEL == KSMD || MODEL == KSCMD )
	double* ecm; //ecm density
	double* ecm_intm;
	double* ecm_rows;
        double* ecm_cols;
	double* ecm_ax;
        double* ecm_ay;
        double* ecm_az;
        double* ecm_bx;
        double* ecm_by;
        double* ecm_bz;
        double* ecm_cx;
        double* ecm_cy;
        double* ecm_cz;
	double* del_ecm;//ECM density gradient	
#endif
	double* d_max_block;
        double* dt; //adaptive time step for the explicit part
        double* new_dt; //stores the adaptive time-step in the host
	char* dat;
} DataArray;

//structure to hold experimental data
typedef struct ExpData
{
#if (STUDY == CALIBRATION)
        double* dzero;
        double* dtwo ;
        double* dfive;
        double* dsvn ;
        double* dnine;
        double* dtwlv;
        double* dfrtn;
#endif
} ExpData;

//Initial data for the simulation
typedef struct InitialData
{
#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
	double da; //Diffusion coefficient for species A
	double s ; //Growth coefficient for species A
#endif
#if ( MODEL == KSC || MODEL == KSCMD  || MODEL == TEST)
	double db; //Diffusion of signal factors
	double chi; //Chemotactic constant
	double r ; //Increase of signal factors
#endif
#if ( MODEL == KSMD || MODEL == KSCMD )
	double dc; //Degradation of ECM as diffusion process
	double chi_ecm; //Chemotactic constant that drives cells towards max ecm density
#endif 
//	double ka; //Cell death term
//	double dup; //uptake term 
        double dt_imp; //Timestep for the implicit part
	double cfl;
	double dx; //Grid spacing
	int itmax; //Max number of iteration steps
	double param; //Parameter for the error (sigma)
	int   dev   ; //GPU device
	int   pid   ; //PID in CPU
        double SSE  ; //Sum squared error between experiment and simulation
#if ( STUDY == SA )	
	double tot_num;
        double bot_num;
#endif
} InitialData;


//Kernel configuration parameter for boundary conditions. All dimensions must be multiple of this.
#define BLOCKSIZE 16
#define BDIMX 16
#define BDIMY 16
#define BDIMZ 16
#define radius 2
#define BXMAX 480
#define BYMAX 480
#define BZMAX 176

//3D Laplace-computing kernel configuration parameters. Dimensions of a kernel is how many threads it contains
#define CELLW	8	//width of thread block
#define CELLH	8	//height of thred block
#define CELLD	8	//depth of thread block. At least 3, but value is arbitrary since CELLW*CELLH is already multiple of the warp size (32)
//NOTE: On devices with compute capability 1.0 and 1.1 (1st generation devices) CELLW should be 16, CELLH and CELLD: 5.
//      this causes reading more data in every step, but the read will be coalesced!
//      On 2nd generation devices all global memory operations are coalesced, block dimensions are chosen to
//      minimize read redundancy. 

//fundamental variables for addressing 3D coordinates in 1D arrays
extern int zsize, ysize, total;

//functions in main.cu
void CudaCheck();
void Error(const char* text);

//functions in host_functions.cu
void hostInitialize(DataArray& host);
void hostClose(DataArray& host);
#if (STUDY == CALIBRATION)
void hostExpInit(ExpData& expdat);
void hostCloseExp(ExpData& expdat);
#endif

//functions in device_functions.cu
void gpuClose(DataArray& device1, DataArray& device2);
void gpuStep_explicit(DataArray& device1, DataArray& device2, InitialData& id, double chi_dx, double chi_ecm_dx, int iter, int time_iter);
void gpuStep_implicit(DataArray& device1, DataArray& device2, double cfl_a, double cfl_b, double cfl_c);
void gpuInitialize(InitialData& id, DataArray& host, DataArray& device1, DataArray& device2);
void ExportDT( DataArray& host, DataArray& device1 );
void ExportCheckpoint(const char* name_a, const char* name_b, const char* name_c, DataArray& host, DataArray& device1, float l, int printout);
//void gpuKa(InitialData& id);

//functions for Monte Carlo
double simulate(DataArray& host, DataArray& device1, DataArray& device2, ExpData& expdat, InitialData& id);

//function to set gpu device
void setDevice(InitialData& id, int tot_gpus);

//funtion to read the data
void readData(InitialData& id, char* argv);

//function to save the data
template <typename T>
void SwapEnd(T& var);
void VTKsave(char fname[4096], int xmax, int ymax, int zmax, double* a);


#if (STUDY == CALIBRATION)
//Calculate SD for NRMSE
#ifdef NRMSE
double calc_ExpSd(double* expdata);
FILE*  save_nrmse(DataArray& host);
#endif
//Calculate Residuals
void calcResiduals(int c, double se, InitialData& id, DataArray& host, ExpData& expdat, FILE* fnrmse);
//Message for status of simulation
void status(int st);
#endif

//Print message
void print_message(InitialData& id, double sigma_data2);

#if (STUDY == SA)
//Calculate responses
void calcResponse(InitialData& id, DataArray& host);
#endif
void test_CD(DataArray& device1, DataArray& device2, InitialData& id);
#endif //declarations_h
