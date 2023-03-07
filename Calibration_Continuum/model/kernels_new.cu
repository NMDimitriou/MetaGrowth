/*
 * Kernels for numerical methods
 * Default kernels Lax-Wendroff-MUSCL for advection
 * and ADI Douglas-Gunn for diffusion.
 * BCs and 19-point centered difference adopted from
 * Molnar et al. DOI: 10.1016/j.chemolab.2011.03.009
 * Nikolaos Dimitriou, McGill, 2021
*/

//Constants for the simulation defined in device Constant memory
__constant__ double DX, _DX	        ; //Same as in InitialData structure
__constant__ int   XMAX, YMAX, ZMAX, NTOT 	; //Dimensions of simulated space and time interval
__constant__ int   Zsize, Ysize              	; //Helps to address memory.
__constant__ double _dt;//, CHI_DX, Da, Db	; //dt is fixed for the diffusion

//Noflux boundary, perpendicular to X axis (YZ sides of space)
__global__ void kernelBoundaryX(double *a_old)
{
	int y=blockIdx.x*BLOCKSIZE+threadIdx.x;
	int z=blockIdx.y*BLOCKSIZE+threadIdx.y;
	int left_dest, left_src, right_dest, right_src;

	left_dest=z*Zsize+y*Ysize;
	left_src=z*Zsize+y*Ysize+1;
	right_dest=z*Zsize+y*Ysize+(XMAX-1);
	right_src=z*Zsize+y*Ysize+(XMAX-2);

	a_old[left_dest]=a_old[left_src];
	a_old[right_dest]=a_old[right_src];
}

//Noflux boundary, perpendicular to Y axis (XZ sides of space)
__global__ void kernelBoundaryY(double *a_old)
{
	int x=blockIdx.x*BLOCKSIZE+threadIdx.x;
	int z=blockIdx.y*BLOCKSIZE+threadIdx.y;
	int left_dest, left_src, right_dest, right_src;

	left_dest=z*Zsize+x;
	left_src=z*Zsize+Ysize+x;
	right_dest=z*Zsize+(YMAX-1)*Ysize+x;
	right_src=z*Zsize+(YMAX-2)*Ysize+x;

	a_old[left_dest]=a_old[left_src];
	a_old[right_dest]=a_old[right_src];
}

//Noflux boundary, perpendicular to Z axis (XY sides of space)
__global__ void kernelBoundaryZ(double *a_old)
{
	int x=blockIdx.x*BLOCKSIZE+threadIdx.x;
	int y=blockIdx.y*BLOCKSIZE+threadIdx.y;
	int left_dest, left_src, right_dest, right_src;

	left_dest=y*Ysize+x;
	left_src=Zsize+y*Ysize+x;
	right_dest=(ZMAX-1)*Zsize+y*Ysize+x;
	right_src=(ZMAX-2)*Zsize+y*Ysize+x;

	a_old[left_dest] =a_old[left_src ];
	a_old[right_dest]=a_old[right_src];
}


//Noflux boundary on edges of space, parallel to X axis
__global__ void kernelBoundaryEdgeX(double *a_old)
{
	int x=blockIdx.x*BLOCKSIZE+threadIdx.x;
	int dest_1, dest_2, dest_3, dest_4;
	int src_1, src_2, src_3, src_4;

	dest_1=x;
	src_1=Zsize+Ysize+x;
	dest_2=(YMAX-1)*Ysize+x;
	src_2=Zsize+(YMAX-2)*Ysize+x;
	dest_3=(ZMAX-1)*Zsize+(YMAX-1)*Ysize+x;
	src_3=(ZMAX-2)*Zsize+(YMAX-2)*Ysize+x;
	dest_4=(ZMAX-1)*Zsize+x;
	src_4=(ZMAX-2)*Zsize+Ysize+x;

	a_old[dest_1]=a_old[src_1];
	a_old[dest_2]=a_old[src_2];
	a_old[dest_3]=a_old[src_3];
	a_old[dest_4]=a_old[src_4];
}


//Noflux boundary on edges of space, parallel to Y axis
__global__ void kernelBoundaryEdgeY(double *a_old)
{
	int y=blockIdx.x*BLOCKSIZE+threadIdx.x;
	int dest_1, dest_2, dest_3, dest_4;
	int src_1, src_2, src_3, src_4;

	dest_1=y*Ysize;
	src_1=Zsize+y*Ysize+1;
	dest_2=y*Ysize+(XMAX-1);
	src_2=Zsize+y*Ysize+(XMAX-2);
	dest_3=(ZMAX-1)*Zsize+y*Ysize+(XMAX-1);
	src_3=(ZMAX-2)*Zsize+y*Ysize+XMAX-2;
	dest_4=(ZMAX-1)*Zsize+y*Ysize;
	src_4=(ZMAX-2)*Zsize+y*Ysize+1;

	a_old[dest_1]=a_old[src_1];
	a_old[dest_2]=a_old[src_2];
	a_old[dest_3]=a_old[src_3];
	a_old[dest_4]=a_old[src_4];
}


//Noflux boundary on edges of space, parallel to Z axis
__global__ void kernelBoundaryEdgeZ(double *a_old)
{
	int z=blockIdx.x*BLOCKSIZE+threadIdx.x;
	int dest_1, dest_2, dest_3, dest_4;
	int src_1, src_2, src_3, src_4;

	dest_1=z*Zsize;
	src_1=z*Zsize+1*Ysize+1;
	dest_2=z*Zsize+(XMAX-1);
	src_2=z*Zsize+Ysize+(XMAX-2);
	dest_3=z*Zsize+(YMAX-1)*Ysize+(XMAX-1);
	src_3=z*Zsize+(YMAX-2)*Ysize+(XMAX-2);
	dest_4=z*Zsize+(YMAX-1)*Ysize;
	src_4=z*Zsize+(YMAX-2)*Ysize+1;

	a_old[dest_1]=a_old[src_1];
	a_old[dest_2]=a_old[src_2];
	a_old[dest_3]=a_old[src_3];
	a_old[dest_4]=a_old[src_4];
}

//boundary perpendicular to X axis (YZ sides of space)
__global__ void kernelBoundaryX_Dirichlet(double *a_old)
{
    int y=blockIdx.x*BLOCKSIZE+threadIdx.x;
    int z=blockIdx.y*BLOCKSIZE+threadIdx.y;
    int left_dest, right_dest;

    left_dest=z*Zsize+y*Ysize;
    right_dest=z*Zsize+y*Ysize+(XMAX-1);

    a_old[left_dest]=0.0;
    a_old[right_dest]=0.0;
}

//boundary perpendicular to Y axis (XZ sides of space)
__global__ void kernelBoundaryY_Dirichlet(double *a_old)
{
   int x=blockIdx.x*BLOCKSIZE+threadIdx.x;
   int z=blockIdx.y*BLOCKSIZE+threadIdx.y;
   int left_dest, right_dest;

    left_dest=z*Zsize+x;
    right_dest=z*Zsize+(YMAX-1)*Ysize+x;

    a_old[left_dest]=0.0;
    a_old[right_dest]=0.0;
}

//boundary perpendicular to Z axis (XY sides of space)
__global__ void kernelBoundaryZ_Mix(double *a_old) //Dirichlet on the upper boundary, Neumann on the lower boundary
{
    int x=blockIdx.x*BLOCKSIZE+threadIdx.x;
    int y=blockIdx.y*BLOCKSIZE+threadIdx.y;
    int left_dest, left_src, right_dest;//, right_src;

    left_dest=y*Ysize+x;
    left_src=Zsize+y*Ysize+x;
    right_dest=(ZMAX-1)*Zsize+y*Ysize+x;
//        right_src=(ZMAX-2)*Zsize+y*Ysize+x;

    a_old[left_dest] =a_old[left_src ];
    a_old[right_dest]=0.0;
}

//boundary on edges of space, parallel to X axis
__global__ void kernelBoundaryEdgeX_Mix(double *a_old)
{
    int x=blockIdx.x*BLOCKSIZE+threadIdx.x;
    int dest_1, dest_2, dest_3, dest_4;
    int src_1, src_2;//, src_3, src_4;

    dest_1=x;
    src_1=Zsize+Ysize+x;
    dest_2=(YMAX-1)*Ysize+x;
    src_2=Zsize+(YMAX-2)*Ysize+x;
    dest_3=(ZMAX-1)*Zsize+(YMAX-1)*Ysize+x;
//  src_3=(ZMAX-2)*Zsize+(YMAX-2)*Ysize+x;
    dest_4=(ZMAX-1)*Zsize+x;
//  src_4=(ZMAX-2)*Zsize+Ysize+x;

    a_old[dest_1]=a_old[src_1];
    a_old[dest_2]=a_old[src_2];
    a_old[dest_3]=0.0;//a_old[src_3];
    a_old[dest_4]=0.0;//a_old[src_4];
}


//boundary on edges of space, parallel to Y axis
__global__ void kernelBoundaryEdgeY_Mix(double *a_old)
{
    int y=blockIdx.x*BLOCKSIZE+threadIdx.x;
    int dest_1, dest_2, dest_3, dest_4;
    int src_1, src_2;//, src_3, src_4;

    dest_1=y*Ysize;
    src_1=Zsize+y*Ysize+1;
    dest_2=y*Ysize+(XMAX-1);
    src_2=Zsize+y*Ysize+(XMAX-2);
    dest_3=(ZMAX-1)*Zsize+y*Ysize+(XMAX-1);
//  src_3=(ZMAX-2)*Zsize+y*Ysize+XMAX-2;
    dest_4=(ZMAX-1)*Zsize+y*Ysize;
//  src_4=(ZMAX-2)*Zsize+y*Ysize+1;

    a_old[dest_1]=a_old[src_1];
    a_old[dest_2]=a_old[src_2];
    a_old[dest_3]=0.0;//a_old[src_3];
    a_old[dest_4]=0.0;//a_old[src_4];
}

//boundary on edges of space, parallel to Z axis
__global__ void kernelBoundaryEdgeZ_Mix(double *a_old)
{
    int z=blockIdx.x*BLOCKSIZE+threadIdx.x;
    int dest_1, dest_2, dest_3, dest_4;
//  int src_1, src_2, src_3, src_4;

    dest_1=z*Zsize;
//  src_1=z*Zsize+1*Ysize+1;
    dest_2=z*Zsize+(XMAX-1);
//  src_2=z*Zsize+Ysize+(XMAX-2);
    dest_3=z*Zsize+(YMAX-1)*Ysize+(XMAX-1);
//  src_3=z*Zsize+(YMAX-2)*Ysize+(XMAX-2);
    dest_4=z*Zsize+(YMAX-1)*Ysize;
//  src_4=z*Zsize+(YMAX-2)*Ysize+1;

  a_old[dest_1]=0.0;//a_old[src_1];
  a_old[dest_2]=0.0;//a_old[src_2];
  a_old[dest_3]=0.0;//a_old[src_3];
  a_old[dest_4]=0.0;//a_old[src_4];

	a_old[0] 		       = a_old[1*Ysize+1];
	a_old[XMAX-1]                  = a_old[Ysize+(XMAX-2)];
	a_old[(YMAX-1)*Ysize+(XMAX-1)] = a_old[(YMAX-2)*Ysize+(XMAX-2)];
	a_old[(YMAX-1)*Ysize]          = a_old[(YMAX-2)*Ysize+1];
}


__device__ int sign(double x)
{
  int t = x<0 ? -1 : 0;
  return x > 0 ? 1 : t;
}


// Growth kernel for species A (cancer cells)
__global__ void kernelGrowA(double *a_new, double *a_old, double *dt, double S)
{

	__shared__ double sa[BDIMY][BDIMX];


	double DT   = *dt; //dt
	double dtS  = DT*S;
//  double dtR  = DT*R;

	int ix = blockIdx.x*blockDim.x + threadIdx.x;
  	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	
//	int in_idx  = iy*XMAX + ix;     //index for reading input
    int out_idx ;           //index for writing output
	int tx = threadIdx.x; //thread's x-index into corresponding shared memory
    int ty = threadIdx.y; //thread's y-index into corresponding shared memory

	#pragma unroll 4
	for(int i=0; i<ZMAX; i++)
  	{
		out_idx = i*Zsize + iy*XMAX + ix;
		sa[ty][tx] = a_old[out_idx];
		//a_new[out_idx] = a_old[out_idx] + dtS*a_old[out_idx]*(1.0-a_old[out_idx]);
		a_new[out_idx] = sa[ty][tx] + dtS*sa[ty][tx]*(1.0-sa[ty][tx]);
	}
}


// Growth kernel for species A (cancer cells)
__global__ void kernelAleeA(double *a_new, double *a_old, double *a_first, double *dt, double S, int iter)
{

    __shared__ double sa[BDIMY][BDIMX];
	__shared__ double sf[BDIMY][BDIMX];


    double DT   = *dt; //dt
    double dtS  = DT*S;
	double time = DT*iter;

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    
    int out_idx ;           //index for writing output
    int tx = threadIdx.x; //thread's x-index into corresponding shared memory
    int ty = threadIdx.y; //thread's y-index into corresponding shared memory

    #pragma unroll 4
    for(int i=0; i<ZMAX; i++)
    {
        out_idx = i*Zsize + iy*XMAX + ix;
        sa[ty][tx] = a_old[out_idx];
		sf[ty][tx] = a_first[out_idx];
        a_new[out_idx] = sa[ty][tx] + dtS*sa[ty][tx]*(1.0  -  1.028*sf[ty][tx]/(sa[ty][tx] + 942.8*time) );
    }
}




// Growth kernel for species B (chemotactic signals)
__global__ void kernelGrowB(double *b_new, double *b_old, double *a_old, double *dt, double R)
{
  	double DT   = *dt; //dt
	double dtR  = DT*R;

  	int ix = blockIdx.x*blockDim.x + threadIdx.x;
  	int iy = blockIdx.y*blockDim.y + threadIdx.y;
  	int out_idx ;

	#pragma unroll 4
  	for(int i=0; i<ZMAX; i++)
  	{
    	out_idx = i*Zsize + iy*XMAX + ix;
   		b_new[out_idx] = b_old[out_idx] + dtR*b_old[out_idx]*a_old[out_idx]*(1.0-a_old[out_idx]);
  	}	
}


// Flux limiter
__device__ double MUSCL_FL(double phi, double r, double aUL, double aUR, double aLL, double aLR)
{
	r   = (aUL - aUR)/(aLL - aLR);
	phi = max(0.0, min(min(0.5*(1.0+r),2.0),2.0*r));
	return phi;
}

// Lax Wendroff step
__device__ double LW(double ap, double a, double am, double sap, double sa, double sam,
										 double bp, double b, double bm, double bmm,
										 double db_p, double db, double db_m,
										 double maxbp1m, double maxbp, double maxbm, double maxbm1p,
										 double LW_term, double CHI_DX, double phip, double phim,
									   double Fp, double Fm, double da)
{
	sap = 1.0-ap;
  	sa  = 1.0-a;
  	sam = 1.0-am;

	db   = CHI_DX*(b  - bm );
	db_m = CHI_DX*(bm - bmm);
	db_p = CHI_DX*(bp - b  );

	maxbp1m = max(0.0,-db_p) ;
	maxbp   = max(0.0, db)   ;
	maxbm   = max(0.0,-db)   ;
	maxbm1p = max(0.0, db_m) ;

	Fp = _DX*(a *sa *maxbp   - ap*sap*maxbp1m + phip*LW_term*(ap*sap*db_p - a *sa *db  ));
  	Fm = _DX*(am*sam*maxbm1p - a *sa *maxbm   + phim*LW_term*(a *sa *db   - am*sam*db_m));

	da = Fm - Fp;
	return da;
}

__global__ void kernelAdv(double *a_new, double *a_old, double *b_old, double *del_b, double *dt, double CHI, double CHI_DX)
{
	__shared__ double sa[BDIMY + 2*radius][BDIMX + 2*radius];
	__shared__ double sb[BDIMY + 2*radius][BDIMX + 2*radius];

	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	int in_idx  = iy*XMAX + ix;  	//index for reading input
	int out_idx ;			//index for writing output

	double DT   = *dt;              //dt
  	double dt_dx= DT*_DX;

	double a_f1, a_f2, a_b1, a_b2;
	double b_f1, b_f2, b_b1, b_b2;
	double a_curr, b_curr;
	double db_m, db_p, db, r;
	double bp1m, bp, bm, bm1p;
	double dax, day, daz, dbx, dby, dbz;
	double phipx, phimx, phipy, phimy, phipz, phimz;
	double Fp, Fm;
	double LW_termx, LW_termy, LW_termz;
	double s, sp, sm;
	int Jx, Jy, sdbx, sdby, sdbz;


	int tx = threadIdx.x + radius; //thread's x-index into corresponding shared memory
	int ty = threadIdx.y + radius; //thread's y-index into corresponding shared memory

	#pragma unroll 5
	for(int i=radius-1; i<=ZMAX-radius; i++)
	{
		if(i == radius-1)
		{
			// fill the "in-front" and "behind" data
			a_b1   = a_old[in_idx]; b_b1   = b_old[in_idx]; in_idx += Zsize;
			a_curr = a_old[in_idx]; b_curr = b_old[in_idx]; out_idx = in_idx; in_idx += Zsize;
			a_f1   = a_old[in_idx]; b_f1   = b_old[in_idx]; in_idx += Zsize;
			a_f2   = a_old[in_idx]; b_f2   = b_old[in_idx]; in_idx += Zsize;

			// Compute ghost points for a_b2, b_b2
			a_b2 = a_curr;
			b_b2 = b_curr; //Neumann B.C.
		}
		else if(i == ZMAX-radius)
		{
			a_b2   = a_b1  ; b_b2   = b_b1;
	    a_b1   = a_curr; b_b1   = b_curr;
	    a_curr = a_f1  ; b_curr = b_f1;
	    a_f1   = a_f2  ; b_f1   = b_f2;
	    a_f2   = a_curr; b_f2   = b_curr; // Ghost points with Neumann BC

	    in_idx  += Zsize;
	    out_idx += Zsize;
		}
		else
		{	// advance the slice (move the thread-front)
			a_b2   = a_b1         ; b_b2   = b_b1;
			a_b1   = a_curr       ; b_b1   = b_curr;
			a_curr = a_f1         ; b_curr = b_f1;
			a_f1   = a_f2  	      ; b_f1   = b_f2;
			a_f2   = a_old[in_idx]; b_f2   = b_old[in_idx];

			in_idx  += Zsize;
			out_idx += Zsize;
		}
		__syncthreads();

		// update the data slice in smem
		if(threadIdx.y<radius) //halo above/below
		{
			sa[threadIdx.y][tx] 		 = a_old[out_idx-radius*XMAX];
			sb[threadIdx.y][tx] 		 = b_old[out_idx-radius*XMAX];
			sa[threadIdx.y+BDIMY+radius][tx] = a_old[out_idx+BDIMY*XMAX];
			sb[threadIdx.y+BDIMY+radius][tx] = b_old[out_idx+BDIMY*XMAX];
		}
		if(threadIdx.x<radius) //halo left/right
		{
			sa[ty][threadIdx.x] 		 = a_old[out_idx-radius];
			sb[ty][threadIdx.x] 		 = b_old[out_idx-radius];
			sa[ty][threadIdx.x+BDIMX+radius] = a_old[out_idx+BDIMX];
			sb[ty][threadIdx.x+BDIMX+radius] = b_old[out_idx+BDIMX];
		}
		__syncthreads();
		//update the slice in smem
		sa[ty][tx] = a_curr;
		sb[ty][tx] = b_curr;
		__syncthreads();

		//compute the output
		dbx = CHI_DX*(sb[ty][tx] - sb[ty][tx-1]);
		dby = CHI_DX*(sb[ty][tx] - sb[ty-1][tx]);
		dbz = CHI_DX*(sb[ty][tx] - b_b1);

		//Sign of db_x,y,z
	  sdbx = sign(dbx);
    sdby = sign(dby);
		sdbz = sign(dbz);

    //Indices for flux limiters
    Jx = tx-sdbx;
    Jy = ty-sdby;

		//MUSCL FL
		phipx = MUSCL_FL(phipx, r, sa[ty][Jx+1], sa[ty][Jx  ], sa[ty][tx+1], sa[ty][tx  ]);
		phimx = MUSCL_FL(phimx, r, sa[ty][Jx  ], sa[ty][Jx-1], sa[ty][tx  ], sa[ty][tx-1]);
		phipy = MUSCL_FL(phipy, r, sa[Jy+1][tx], sa[Jy  ][tx], sa[ty+1][tx], sa[ty  ][tx]);
		phimy = MUSCL_FL(phimy, r, sa[Jy  ][tx], sa[Jy-1][tx], sa[ty  ][tx], sa[ty-1][tx]);

		//Lax-Wendroff
	  LW_termx = 0.5*(sdbx-sdbx*CHI*dt_dx);
	  LW_termy = 0.5*(sdby-sdby*CHI*dt_dx);
	  LW_termz = 0.5*(sdbz-sdbz*CHI*dt_dx);

		dax = LW(sa[ty][tx+1], sa[ty][tx], sa[ty][tx-1], sp, s, sm,
						 sb[ty][tx+1], sb[ty][tx], sb[ty][tx-1], sb[ty][tx-2],
						 db_p, db, db_m, bp1m, bp, bm, bm1p,
						 LW_termx, CHI_DX, phipx, phimx, Fp, Fm, dax);

		day = LW(sa[ty+1][tx], sa[ty][tx], sa[ty-1][tx], sp, s, sm,
						 sb[ty+1][tx], sb[ty][tx], sb[ty-1][tx], sb[ty-2][tx],
						 db_p, db, db_m, bp1m, bp, bm, bm1p,
						 LW_termy, CHI_DX, phipy, phimy, Fp, Fm, day);


		if(dbz > 0.0)
		{
				phipz = MUSCL_FL(phipz, r, sa[ty][tx], a_b1, a_f1      , sa[ty][tx]);
				phimz = MUSCL_FL(phimz, r, a_b1      , a_b2, sa[ty][tx], a_b1      );
	  }
	  else
	 	{
				phipz = MUSCL_FL(phipz, r, a_f2, a_f1      , a_f1      , sa[ty][tx]);
				phimz = MUSCL_FL(phimz, r, a_f1, sa[ty][tx], sa[ty][tx], a_b1			 );
		}

		daz = LW(a_f1, sa[ty][tx], a_b1, sp, s, sm,
						 b_f1, sb[ty][tx], b_b1, b_b2,
						 db_p, db, db_m, bp1m, bp, bm, bm1p,
						 LW_termz, CHI_DX, phipz, phimz, Fp, Fm, daz);


		del_b[out_idx] = dbx + dby + dbz;

		//update
	  a_new[out_idx] = sa[ty][tx] + DT*(dax+day+daz);

	}//for(int i=radius-1; i<=ZMAX-radius; i++)
} // kernelAdv


// Thomas algorithm
__device__ void thomas(double *a, double *b, double *c, double *x, double *z, int sizeSmallerSystem, int stride){

        int systemSize = stride*sizeSmallerSystem;
        int i = threadIdx.x;
        int startLocationSystem, endLocationSystem;
        double tmp;

        c[i] = c[i] / b[i];
        z[i] = z[i] / b[i];
        startLocationSystem = stride + i;
        for (int i = startLocationSystem;i<systemSize-1;i += stride){ //was i<systemSize
            tmp = b[i]-a[i]*c[i-stride];
            c[i]  =  c[i] / tmp;
            z[i]  = (z[i]-z[i-stride]*a[i]) / tmp;
        }
        endLocationSystem = systemSize-stride + i;
        x[endLocationSystem] = z[endLocationSystem];
        for (int i = endLocationSystem-stride;i>= 1;i-= stride) x[i] = z[i]-c[i]*x[i + stride]; //printf("x = %lf\n", x[i]); was i>-0

}


// Hybrid Parallel Cyclic Reduction - Thomas algorighm
__device__ void PCRTHOMASglobal(double *a, double *b, double *c, double *x, double *z, int numSteps, int sizeSystem, int sizeSmallerSystem) {


	int delta = 1;
 	int thomasstride;
  	int iLeft, iRight;
  	double tmp1, tmp2, bNew, zNew, aNew, cNew;
  	double bNew1, zNew1, aNew1, cNew1;

	int i = threadIdx.x;
  	int i1 = i+sizeSystem/2;
  	for (int j = 0; j < numSteps; j++) {
    	iRight = i+delta;
    	iRight = (iRight>=sizeSystem-1)?sizeSystem-2:iRight; //sizeSystem, sizeSystem-1
    	iLeft = i-delta;
  		iLeft = (iLeft <1)?1:iLeft; //0,0

    	tmp1 = a[i] / b[iLeft];
    	tmp2 = c[i] / b[iRight];
  		bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
    	zNew = z[i] - z[iLeft] * tmp1 - z[iRight] * tmp2;
  		aNew = -a[iLeft ] * tmp1;
  		cNew = -c[iRight] * tmp2;

		iRight = i1+delta;
		iRight = (iRight>=sizeSystem-1)?sizeSystem-2:iRight;
		iLeft  = i1-delta;
		iLeft  = (iLeft <1)?1:iLeft;
		tmp1   = a[i1] / b[iLeft];
		tmp2   = c[i1] / b[iRight];
		bNew1  = b[i1] - c[iLeft] * tmp1 - a[iRight] * tmp2;
		zNew1  = z[i1] - z[iLeft] * tmp1 - z[iRight] * tmp2;
		aNew1  = -a[iLeft ] * tmp1;
		cNew1  = -c[iRight] * tmp2;

		__syncthreads();
		b[i ] = bNew;
		z[i ] = zNew;
		a[i ] = aNew;
		c[i ] = cNew;
		b[i1] = bNew1;
		z[i1] = zNew1;
		a[i1] = aNew1;
		c[i1] = cNew1;
		__syncthreads();
		delta *= 2;
	}
	thomasstride = sizeSystem/sizeSmallerSystem;

	if (i < thomasstride)   thomas(a,b,c,x,z,sizeSmallerSystem,thomasstride);
}

__global__ void transpose_XYZ_to_ZXY(double *osol, double *sol)
{
	__shared__ double block[BLOCKSIZE][BLOCKSIZE + 1];

	int xin, y, zin, xout, zout, iid, oid;

	int tx = threadIdx.x,
			ty = threadIdx.y,
      bx = blockIdx.x,
      by = blockIdx.y;

	xin = tx + BLOCKSIZE * bx;
	zin = ty + BLOCKSIZE * by;

	y = blockIdx.z;

  	xout = ty + BLOCKSIZE * bx;
  	zout = tx + BLOCKSIZE * by;


	iid = xin  + (y +  zin * YMAX) * XMAX;
	oid = zout + (xout + y * XMAX) * ZMAX;

	if( xin < XMAX && zin < ZMAX )
		block[tx][ty] = sol[iid];

	__syncthreads();

	if( xout < XMAX && zout < ZMAX )
		osol[oid] = block[ty][tx];
}

__global__ void transpose_ZXY_to_XYZ(double *osol, double *sol)
{

	__shared__ double block[BLOCKSIZE][BLOCKSIZE + 1];

	int xin, yin, z, xout, yout, iid, oid;

	int tx = threadIdx.x,
	ty = threadIdx.y,
    bx = blockIdx.x,
    by = blockIdx.y;

	xin = tx + BLOCKSIZE * bx;
	yin = ty + BLOCKSIZE * by;

	z = blockIdx.z;

	xout = ty + BLOCKSIZE * bx;
	yout = tx + BLOCKSIZE * by;

	iid = xin  + (yin +    z * XMAX) * ZMAX;
	oid = yout + (z   + xout * YMAX) * XMAX;

	if( xin < ZMAX && yin < XMAX )
		block[tx][ty] = sol[iid];


	__syncthreads();

	if( yout < XMAX && xout < ZMAX )
		osol[oid] = block[ty][tx];

}


__global__ void fillTridiagonals_x(double *d_a, double *d_b, double *d_c, double *sol, double *sol_rows, int elements, double DuDt_dxdx)
{
	__shared__ double aa[BXMAX];
  	__shared__ double bb[BXMAX];
  	__shared__ double cc[BXMAX];
  	__shared__ double zz[BXMAX];
  	__shared__ double xr[BXMAX];

  	int idx = blockIdx.x;
  	int idy = blockIdx.y;
  	int ty  = threadIdx.y;
  	int k   = blockDim.y*idy+ty;
  	int j, indJ;
  	bool ok_compute = (idx>0 && idx<YMAX-1 && k>0 && k<ZMAX-1);

  if(ok_compute)
  {
	int index = idx*XMAX;
	for(int e = 0; e < elements; e++){
		j       =   elements*threadIdx.x+e;
  		indJ    = k*XMAX*YMAX + index + j;

      	bb[j]     = 1.0+DuDt_dxdx;
		aa[j]     =-0.5*DuDt_dxdx;
      	cc[j]     =-0.5*DuDt_dxdx;
		zz[j]     = (j>0 && j<XMAX-1) ? (DuDt_dxdx*( 0.5* sol[indJ - 1] + 0.5*sol[indJ + 1]
                    + sol[indJ - XMAX	] +     sol[indJ + XMAX ]
                  	+ sol[indJ - XMAX*YMAX	] +     sol[indJ + XMAX*YMAX])
                  	+ (1.0 - 5.0*DuDt_dxdx)*sol[indJ]) : 0.0f;

		xr[j] = sol_rows[indJ];	
	}
	__syncthreads();
	PCRTHOMASglobal(aa,bb,cc,xr,zz,7,XMAX,8); //7,8
	__syncthreads();

	for(int e = 0; e < elements; e++)
	{
		j           = elements*threadIdx.x+e;
		indJ        = k*XMAX*YMAX + index + j;
		sol_rows[indJ]  = xr[j];
	}
	__syncthreads();	
  }
}

__global__ void fillTridiagonals_y(double *d_a, double *d_b, double *d_c, double *sol, double *sol_rows, double *sol_cols, int elements, double DuDt_dxdx){

	__shared__ double aa[BYMAX];
	__shared__ double bb[BYMAX];
	__shared__ double cc[BYMAX];
	__shared__ double zz[BYMAX];
	__shared__ double xc[BYMAX];

	int idx = blockIdx.x;
	int idy = blockIdx.y;
	int ty  = threadIdx.y;
	int k   = blockDim.y*idy+ty;
	int j, ind;
	bool ok_compute = (idx>0 && idx<XMAX-1 && k>0 && k<ZMAX-1);

	if(ok_compute){
		for(int e = 0; e < elements; e++)
		{
			j       = elements*threadIdx.x+e;
			ind     = k*XMAX*YMAX + idx   + j*YMAX;

			bb[j] =  1.0+DuDt_dxdx ;
			aa[j] = -0.5*DuDt_dxdx ;
			cc[j] = -0.5*DuDt_dxdx ;
			zz[j] = (j>0 && j<YMAX-1) ? (-0.5*DuDt_dxdx*(sol[ind-XMAX] + sol[ind+XMAX] - 2.0*sol[ind])+sol_rows[ind]) : 0.0f;
							xc[j] = sol_cols[ind];


	 	}
		__syncthreads();
		PCRTHOMASglobal(aa,bb,cc,xc,zz,7,YMAX,8);
		__syncthreads();

		for(int e = 0; e < elements; e++)
		{
			j     =   elements*threadIdx.x+e;
			ind   = k*XMAX*YMAX + idx   + j*YMAX;
			sol_cols[ind] = xc[j];
		}
		__syncthreads();
	}
}


__global__ void fillTridiagonals_z_transp(double *d_a, double *d_b, double *d_c, double *sol_cols, double *osol, int elements, double DuDt_dxdx){

	__shared__ double aa[BZMAX];
	__shared__ double bb[BZMAX];
	__shared__ double cc[BZMAX];
  	__shared__ double zz[BZMAX];
  	__shared__ double xx[BZMAX];

	int idx = blockIdx.x;
	int idy = blockIdx.y;
	int ty  = threadIdx.y;
	int k   = blockDim.y*idy+ty;
	int j, ind, indJ;
	bool ok_compute = (idx>0 && idx<XMAX-1 && k>0 && k<YMAX-1);

	if(ok_compute){
		int index = idx*ZMAX;
		for(int e = 0; e < elements; e++)
		{
    	j    = elements*threadIdx.x+e;
			ind  = k*XMAX + idx + j*XMAX*YMAX; //k updates y-dim, idx updates x-dim, j updates z-dim
//    indJ = k*NX*NZ + idx + j*NX; //indices for the transposed 3d matrix xyz->xzy
			indJ = k*XMAX*ZMAX + index + j; //indices for the transposed 3d matrix xyz->zxy

			bb[j] = 1.0+DuDt_dxdx;
			aa[j] =-0.5*DuDt_dxdx;
			cc[j] =-0.5*DuDt_dxdx;
//    zz[j] = (j>0 && j< 175) ? (-0.5*DuDt_dxdx*(osol[indJ-NX] + osol[indJ+NX] - 2.0*osol[indJ]) + sol_cols[ind]) : 0.0f;
      zz[j] = (j>0 && j<ZMAX-1) ? (-0.5*DuDt_dxdx*(osol[indJ-1] + osol[indJ+1] - 2.0*osol[indJ]) + sol_cols[ind]) : 0.0f;

      xx[j] = osol[indJ];
		}

		__syncthreads();
		PCRTHOMASglobal(aa,bb,cc,xx,zz,7,ZMAX,8);
		__syncthreads();

		for(int e = 0; e < elements; e++)
		{
			j    = elements*threadIdx.x+e;
//   	ind  = k*NX + idx + j*NX*NY;
//    indJ = k*NX*NZ + idx + j*NX; //indices for the transposed 3d matrix xyz->xzy
			indJ = k*XMAX*ZMAX + index + j; //indices for the transposed 3d matrix xyz->zxy
			osol[indJ]= xx[j];
		}
  	__syncthreads();
	}
	__syncthreads();
}




__device__ void warp_reduce_max( double smem[256])
{
	smem[threadIdx.x] = smem[threadIdx.x+128] > smem[threadIdx.x] ? smem[threadIdx.x+128] : smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+ 64] > smem[threadIdx.x] ? smem[threadIdx.x+ 64] : smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+ 32] > smem[threadIdx.x] ? smem[threadIdx.x+ 32] : smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+ 16] > smem[threadIdx.x] ? smem[threadIdx.x+ 16] : smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+  8] > smem[threadIdx.x] ? smem[threadIdx.x+  8] : smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+  4] > smem[threadIdx.x] ? smem[threadIdx.x+  4] : smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+  2] > smem[threadIdx.x] ? smem[threadIdx.x+  2] : smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+  1] > smem[threadIdx.x] ? smem[threadIdx.x+  1] : smem[threadIdx.x]; __syncthreads();
}

__device__ void warp_reduce_min( double smem[256])
{
  smem[threadIdx.x] = smem[threadIdx.x+128] < smem[threadIdx.x] ? smem[threadIdx.x+128] : smem[threadIdx.x]; __syncthreads();
  smem[threadIdx.x] = smem[threadIdx.x+ 64] < smem[threadIdx.x] ? smem[threadIdx.x+ 64] : smem[threadIdx.x]; __syncthreads();
  smem[threadIdx.x] = smem[threadIdx.x+ 32] < smem[threadIdx.x] ? smem[threadIdx.x+ 32] : smem[threadIdx.x]; __syncthreads();
  smem[threadIdx.x] = smem[threadIdx.x+ 16] < smem[threadIdx.x] ? smem[threadIdx.x+ 16] : smem[threadIdx.x]; __syncthreads();
  smem[threadIdx.x] = smem[threadIdx.x+  8] < smem[threadIdx.x] ? smem[threadIdx.x+  8] : smem[threadIdx.x]; __syncthreads();
  smem[threadIdx.x] = smem[threadIdx.x+  4] < smem[threadIdx.x] ? smem[threadIdx.x+  4] : smem[threadIdx.x]; __syncthreads();
  smem[threadIdx.x] = smem[threadIdx.x+  2] < smem[threadIdx.x] ? smem[threadIdx.x+  2] : smem[threadIdx.x]; __syncthreads();
  smem[threadIdx.x] = smem[threadIdx.x+  1] < smem[threadIdx.x] ? smem[threadIdx.x+  1] : smem[threadIdx.x]; __syncthreads();
}

__global__ void kernelStability(double *del_b, double *del_c, double *max_block)
{

  unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int stride = gridDim.x*blockDim.x;
  unsigned int offset = 0;

  __shared__ double cache_CFL [256]; // Careful!! BLOCKSIZE is also used by BC

  double maxCFL  = -1.0;
  double tmp_CFL;

  while(index + offset < NTOT){

  	maxCFL = max(maxCFL, abs(del_b[index+offset])); //del_b includes CHI
	#if ( MODEL == KSCMD )
	maxCFL = max(maxCFL, abs(del_c[index+offset]));
	#endif
    offset += stride;	
  }

	cache_CFL[threadIdx.x] = maxCFL;
	__syncthreads();

	if(threadIdx.x < 128)
		warp_reduce_max(cache_CFL);

	if(threadIdx.x == 0){
		tmp_CFL = 0.3*DX/cache_CFL[0];
		max_block[blockIdx.x] = tmp_CFL;
  	}
}

__global__ void final_kernelStability(double *max_block, double *DT_old, double *DT, int N, int iter)//, double Da, double Db)
{

	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[256];

	double temp = 1000.0;
	while(index + offset < N){
    temp = min(temp, max_block[index + offset]);
		offset += stride;
	}

	cache[threadIdx.x] = temp;
	__syncthreads();

	if(threadIdx.x < 128)
	{
		warp_reduce_min(cache);
	}

	if(threadIdx.x == 0){
		max_block[blockIdx.x] = cache[0];
		if(iter ==1){
		//	if(Da<=0.0001 && Db <= 0.0001){
                //		DT[blockIdx.x] = min(0.01/_dt,max_block[blockIdx.x]);
		//	}else{
			DT[blockIdx.x] = min(0.1/_dt,max_block[blockIdx.x]);
		//	}
		}else{
			DT[blockIdx.x] = min(DT_old[blockIdx.x], max_block[blockIdx.x]);
		//	if(Da<=0.0001 && Db <= 0.0001){
		//		DT[blockIdx.x] = min(0.01/_dt,DT[blockIdx.x]);
		//	}else{
			DT[blockIdx.x] = min(0.1/_dt,DT[blockIdx.x]);
		//	}
		}
	}
}



//__global__ void kernelDiffusion(double *a_old, double *b_old, double *daxyz, double *a_new, double *b_new, double* dt)
__global__ void kernelDiffusion(double *a_old, double *a_new, double dt, double Da)
{
	__shared__ double ao    [CELLD][CELLH][CELLW*2+1];
//        __shared__ double bo    [CELLD][CELLH][CELLW*2+1];
//	__shared__ double dxyzo [CELLD][CELLH][CELLW*2+1];

  unsigned int p, k, x, y, z;
  bool ok_read, ok_compute;

	double laplace_a;//, laplace_b;
//	double k1a,k2a,k3a,k4a;
//        double k1b,k2b,k3b,k4b;

	double DT   = dt;              //dt
//        double Dt_2, Dt3_2, Dt_6;

//        Dt_2 = DT/2.0;
//        Dt3_2= 3.0*DT/2.0;
//        Dt_6 = DT/6.0;

	//3D position and memory address
        z = blockIdx.y*(CELLD-2)+threadIdx.z;
        y = blockIdx.x*(CELLH-2)+threadIdx.y;
        x = threadIdx.x;
        p = z * Zsize + y * Ysize + x;

	//precompute conditions
  ok_read = (z<ZMAX) & (y<YMAX);
  ok_compute = (threadIdx.y>0) & (threadIdx.y<CELLH-1) & (threadIdx.z>0) & (threadIdx.z<CELLD-1) & (y<YMAX-1) & (z<ZMAX-1);

  //read first two tiles of data
  for (int n=0; n<14; n++){
  	if (!ok_read) {
			break;
		}
		//if (ok_read)
		//{
		ao[threadIdx.z][threadIdx.y][threadIdx.x+1]  = a_old[p];
//  bo[threadIdx.z][threadIdx.y][threadIdx.x+1]  = b_old[p];
//	dxyzo[threadIdx.z][threadIdx.y][threadIdx.x+1] = daxyz[p];
		p+=CELLW;
		ao[threadIdx.z][threadIdx.y][threadIdx.x+1+CELLW]  = a_old[p];
//  bo[threadIdx.z][threadIdx.y][threadIdx.x+1+CELLW]  = b_old[p];
//	dxyzo[threadIdx.z][threadIdx.y][threadIdx.x+1+CELLW] = daxyz[p];
//};

	}
	__syncthreads();

	#pragma unroll 5
  //Move Tiles to the right: computing, writing result and reading new data in each iteration step.
  for (k=0; k < XMAX/CELLW; k++)
  {
		x = k*CELLW + threadIdx.x;
		p = z * Zsize + y * Ysize + x;

		//calculate
		if (ok_compute & (x>0) & (x<XMAX-1))
		{
			// 2nd order 19-point centered differences in space
      laplace_a =
                 	ao[threadIdx.z+1][threadIdx.y-1][threadIdx.x+1] +
               		ao[threadIdx.z+1][threadIdx.y  ][threadIdx.x  ] +
                 	ao[threadIdx.z+1][threadIdx.y  ][threadIdx.x+2] +
             			ao[threadIdx.z+1][threadIdx.y+1][threadIdx.x+1] +
                 	ao[threadIdx.z  ][threadIdx.y-1][threadIdx.x  ] +
               		ao[threadIdx.z  ][threadIdx.y-1][threadIdx.x+2] +
               		ao[threadIdx.z  ][threadIdx.y+1][threadIdx.x  ] +
                 	ao[threadIdx.z  ][threadIdx.y+1][threadIdx.x+2] +
               		ao[threadIdx.z-1][threadIdx.y-1][threadIdx.x+1] +
               		ao[threadIdx.z-1][threadIdx.y  ][threadIdx.x  ] +
                 	ao[threadIdx.z-1][threadIdx.y  ][threadIdx.x+2] +
                 	ao[threadIdx.z-1][threadIdx.y+1][threadIdx.x+1] +
         2.0 *(  	ao[threadIdx.z-1][threadIdx.y  ][threadIdx.x+1] +
             			ao[threadIdx.z+1][threadIdx.y  ][threadIdx.x+1] +
               		ao[threadIdx.z  ][threadIdx.y-1][threadIdx.x+1] +
             			ao[threadIdx.z  ][threadIdx.y+1][threadIdx.x+1] +
             			ao[threadIdx.z  ][threadIdx.y  ][threadIdx.x  ] +
           				ao[threadIdx.z  ][threadIdx.y  ][threadIdx.x+2]
      		) -24.0 *ao[threadIdx.z  ][threadIdx.y  ][threadIdx.x+1];

//				ao[threadIdx.z-1][threadIdx.y  ][threadIdx.x+1] +
//				ao[threadIdx.z+1][threadIdx.y  ][threadIdx.x+1] +
//				ao[threadIdx.z  ][threadIdx.y-1][threadIdx.x+1] +
//                                ao[threadIdx.z  ][threadIdx.y+1][threadIdx.x+1] +
//                                ao[threadIdx.z  ][threadIdx.y  ][threadIdx.x  ] +
//                                ao[threadIdx.z  ][threadIdx.y  ][threadIdx.x+2] -
//			6.0    *ao[threadIdx.z  ][threadIdx.y  ][threadIdx.x+1];


/*			laplace_b =
                                 bo[threadIdx.z+1][threadIdx.y-1][threadIdx.x+1] +
                                 bo[threadIdx.z+1][threadIdx.y  ][threadIdx.x  ] +
                                 bo[threadIdx.z+1][threadIdx.y  ][threadIdx.x+2] +
                                 bo[threadIdx.z+1][threadIdx.y+1][threadIdx.x+1] +
                                 bo[threadIdx.z  ][threadIdx.y-1][threadIdx.x  ] +
                                 bo[threadIdx.z  ][threadIdx.y-1][threadIdx.x+2] +
                                 bo[threadIdx.z  ][threadIdx.y+1][threadIdx.x  ] +
                                 bo[threadIdx.z  ][threadIdx.y+1][threadIdx.x+2] +
                                 bo[threadIdx.z-1][threadIdx.y-1][threadIdx.x+1] +
                                 bo[threadIdx.z-1][threadIdx.y  ][threadIdx.x  ] +
                                 bo[threadIdx.z-1][threadIdx.y  ][threadIdx.x+2] +
                                 bo[threadIdx.z-1][threadIdx.y+1][threadIdx.x+1] +
                         2.0*(   bo[threadIdx.z-1][threadIdx.y  ][threadIdx.x+1] +
                                 bo[threadIdx.z+1][threadIdx.y  ][threadIdx.x+1] +
                                 bo[threadIdx.z  ][threadIdx.y-1][threadIdx.x+1] +
                                 bo[threadIdx.z  ][threadIdx.y+1][threadIdx.x+1] +
                                 bo[threadIdx.z  ][threadIdx.y  ][threadIdx.x  ] +
                                 bo[threadIdx.z  ][threadIdx.y  ][threadIdx.x+2]
                        ) -24.0* bo[threadIdx.z  ][threadIdx.y  ][threadIdx.x+1];

*/

        		a_new[p]=ao[threadIdx.z][threadIdx.y][threadIdx.x+1] + DT*Da*laplace_a/(6.0*DX*DX);//+ Dt_6*(k1a + 2.0f*k2a + 2.0f*k3a + k4a);
//        		b_new[p]=bo[threadIdx.z][threadIdx.y][threadIdx.x+1] + Dt_6*(k1b + 2.0f*k2b + 2.0f*k3b + k4b);

		}

		__syncthreads();

		//copy last column of first tile to the extra column before the first tile (no bank conflict)
                if (threadIdx.x==CELLW-1)
                {
                        ao[threadIdx.z][threadIdx.y][0]  = ao[threadIdx.z][threadIdx.y][CELLW];
//                        bo[threadIdx.z][threadIdx.y][0]  = bo[threadIdx.z][threadIdx.y][CELLW];
//			dxyzo[threadIdx.z][threadIdx.y][0] = dxyzo[threadIdx.z][threadIdx.y][CELLW];
                };

                //no need to syncthreads() here because threads (warps) that read in the memcopy above
                //are exactly the ones that will write to the same address in the following memcopy

                //moving the tile: copy the second tile onto the first
                //no bank conflict -> this is as fast as setting a new value to a register (in every thread)
                ao[threadIdx.z][threadIdx.y][threadIdx.x+1]  = ao[threadIdx.z][threadIdx.y][threadIdx.x+1 + CELLW];
//                bo[threadIdx.z][threadIdx.y][threadIdx.x+1]  = bo[threadIdx.z][threadIdx.y][threadIdx.x+1 + CELLW];
//		dxyzo[threadIdx.z][threadIdx.y][threadIdx.x+1] = dxyzo[threadIdx.z][threadIdx.y][threadIdx.x+1 + CELLW];

                //read new data into the second tile
                if (k < XMAX/CELLW -2) //don't read in last two iterations
                {
                        x = (k+2) * CELLW + threadIdx.x;
                        p = z * Zsize + y * Ysize + x;
                        if (ok_read & (x<XMAX))
                        {
                                ao[threadIdx.z][threadIdx.y][threadIdx.x+1 + CELLW]  = a_old[p];
//                                bo[threadIdx.z][threadIdx.y][threadIdx.x+1 + CELLW]  = b_old[p];
//				dxyzo[threadIdx.z][threadIdx.y][threadIdx.x+1 + CELLW] = daxyz[p];
                        }
                }

                __syncthreads();
        }
}


/*
__device__ void warp_reduce_max( double smem[256])
{
        smem[threadIdx.x] = smem[threadIdx.x+128] > smem[threadIdx.x] ? smem[threadIdx.x+128] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+ 64] > smem[threadIdx.x] ? smem[threadIdx.x+ 64] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+ 32] > smem[threadIdx.x] ? smem[threadIdx.x+ 32] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+ 16] > smem[threadIdx.x] ? smem[threadIdx.x+ 16] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+  8] > smem[threadIdx.x] ? smem[threadIdx.x+  8] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+  4] > smem[threadIdx.x] ? smem[threadIdx.x+  4] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+  2] > smem[threadIdx.x] ? smem[threadIdx.x+  2] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+  1] > smem[threadIdx.x] ? smem[threadIdx.x+  1] : smem[threadIdx.x]; __syncthreads();

}

__device__ void warp_reduce_min( double smem[256])
{
        smem[threadIdx.x] = smem[threadIdx.x+128] < smem[threadIdx.x] ? smem[threadIdx.x+128] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+ 64] < smem[threadIdx.x] ? smem[threadIdx.x+ 64] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+ 32] < smem[threadIdx.x] ? smem[threadIdx.x+ 32] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+ 16] < smem[threadIdx.x] ? smem[threadIdx.x+ 16] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+  8] < smem[threadIdx.x] ? smem[threadIdx.x+  8] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+  4] < smem[threadIdx.x] ? smem[threadIdx.x+  4] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+  2] < smem[threadIdx.x] ? smem[threadIdx.x+  2] : smem[threadIdx.x]; __syncthreads();
        smem[threadIdx.x] = smem[threadIdx.x+  1] < smem[threadIdx.x] ? smem[threadIdx.x+  1] : smem[threadIdx.x]; __syncthreads();

}

__global__ void kernelStability(double *a, double *del_b, double *max_block)
{

        unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned int stride = gridDim.x*blockDim.x;
        unsigned int offset = 0;

//        __shared__ double cache_Nmn [256];
        __shared__ double cache_CFL [256]; // Careful!! BLOCKSIZE is also used by BC
//        __shared__ double cache_Nmn2[256];


        double maxCFL  = -1.0;
        double maxNmn  = max(Da,Db);
//        double minNmn2 = Da;
        double tmp_Nmn;
        double tmp_CFL;

        while(index + offset < NTOT){

//              maxNmn = max(maxNmn,   CHI*a[index + offset] );
		maxCFL = max(maxCFL, abs(del_b[index+offset]));

//                if(CHI*a[index+offset]          > Da/10.0       ){      minNmn2     = min(CHI*a[index+offset], Da);             }

                offset += stride;
        }

//        cache_Nmn[threadIdx.x] = maxNmn;
        cache_CFL[threadIdx.x] = maxCFL;
//        cache_Nmn2[threadIdx.x]= minNmn2;

        __syncthreads();

        if(threadIdx.x < 128)
        {
//                warp_reduce_max(cache_Nmn);
                warp_reduce_max(cache_CFL);
//                warp_reduce_min(cache_Nmn2);
        }
	if(threadIdx.x == 0){

                tmp_Nmn = 0.4*DX*DX/maxNmn;
                tmp_CFL = 0.8*DX/cache_CFL[0];

//		printf("Nmn = %1.1e\nCFL = %1.1e\n Nmn2 = %1.1e\n", tmp_Nmn, tmp_CFL, cache_Nmn2[0]);

		max_block[blockIdx.x] = min(tmp_Nmn, tmp_CFL);
//		max_block[blockIdx.x] = min(max_block[blockIdx.x], cache_Nmn2[0]);
//		max_block[blockIdx.x] = min(max_block[blockIdx.x], tmp_Nmn);
//              DT[blockIdx.x] = fmin(DT[blockIdx.x],2.0*cache_Nmn2[0]/(cache_CFL[0]*cache_CFL[0]));
//		printf("max_block = %1.1e\n", max_block[blockIdx.x]);

        }
}


__global__ void final_kernelStability(double *max_block, double *DT_old, double *DT, int N)
{

        unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned int stride = gridDim.x*blockDim.x;
        unsigned int offset = 0;

        __shared__ double cache[256];


        double temp = 1000.0;
        while(index + offset < N){
                temp = min(temp, max_block[index + offset]);

                offset += stride;
        }

        cache[threadIdx.x] = temp;

        __syncthreads();

        if(threadIdx.x < 128)
        {
                warp_reduce_min(cache);
        }

        if(threadIdx.x == 0){

                max_block[blockIdx.x] = cache[0];
                DT[blockIdx.x] = min(DT_old[blockIdx.x], max_block[blockIdx.x]);//0.4*DX*DX/max_block[blockIdx.x];
//		printf("DT = %1.1e\n", DT[blockIdx.x]);

        }
}
*/
