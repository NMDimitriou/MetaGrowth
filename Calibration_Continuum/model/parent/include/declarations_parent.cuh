#ifndef DECLARATIONS_H
#define DECLARATIONS_H

#include <iostream>
#include <cstdlib>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include<sys/shm.h>
#include<sys/stat.h>
#include <math.h>
#include <algorithm>

using namespace std;

#define NONE            0

#define FK              1
#define KSC             2
#define KSMD            3
#define KSCMD           4
#define TEST		5

#define CALIBRATION     1
#define SA              2

#define EXP             1
#define LIN             2
#define ID              3
#define NA				4

//structure to hold size of simulation and arrays on the host
typedef struct DataArray
{
	int xmax;
	int ymax;
	int zmax;

	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
	double* a; //cell density
	//Parameters for tridiagonal solver
        double* a_rows;
        double* a_cols;
	int segid_a           ;
	int segid_a_rows      ;
        int segid_a_cols      ;
	#endif

	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
	double* b; //signal density
	//Parameters for tridiagonal solver
	double* b_rows; 
	double* b_cols;
	double* del_b; //signal density gradient
	int segid_b           ;
        int segid_b_rows      ;
        int segid_b_cols      ;
        int segid_del_b       ;
	#endif

	#if ( MODEL == KSMD || MODEL == KSCMD)
	double* ecm;
	double* ecm_rows;
        double* ecm_cols;
	double* del_ecm;
	int segid_ecm         ;
        int segid_ecm_rows    ;
        int segid_ecm_cols    ;
        int segid_del_ecm     ;
	#endif

	double* d_max_block;
        int segid_d_max_block ;
	char* dat;
	char* path;

} DataArray;

#if ( STUDY == CALIBRATION )
//structure to hold experimental data
typedef struct ExpData
{
        double* dzero;
        double* dtwo ;
        double* dfive;
        double* dsvn ;
        double* dnine;
        double* dtwlv;
        double* dfrtn;
	
	int segid_dzero;
        int segid_dtwo ;
        int segid_dfive;
        int segid_dsvn ;
        int segid_dnine;
        int segid_dtwlv;
        int segid_dfrtn;

} ExpData;
#endif

//fundamental variables for addressing 3D coordinates in 1D arrays
extern int zsize, ysize, total;

//functions in main.cu
void CudaCheck();
void Error(const char* text);

//functions in host_functions.cu
void hostInitializeParent(DataArray& host, char* hostname);
void hostClose(DataArray& host);
#if ( STUDY == CALIBRATION )
void hostExpInitParent(DataArray& host, ExpData& expdat, char* hostname);
void hostCloseExp(ExpData& expdat);
#endif
#endif //declarations_h
