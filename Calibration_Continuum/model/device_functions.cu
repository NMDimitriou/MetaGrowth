#include "declarations.cuh"
#include "kernels_new.cu"

/* Initialize parameters for GPUs
 * Nikolaos Dimitriou, McGill, 2021
*/

//kernel configuration variables
dim3 blocks_laplace;	
dim3 threads_laplace;

dim3 blocks_adv;
dim3 threads_adv;

dim3 blocksDG_x;
dim3 threadsDG_x;
dim3 blocksDG_y;
dim3 threadsDG_y;
dim3 blocksDG_z;
dim3 threadsDG_z;

dim3 grid_xyz2zxy;
dim3 threads_xyz2zxy;

dim3 grid_zxy2xyz;
dim3 threads_zxy2xyz;

dim3 blocks_boundary_x;
dim3 blocks_boundary_y;
dim3 blocks_boundary_z;
dim3 threads_boundary;

dim3 blocks_edge_x;
dim3 blocks_edge_y;
dim3 blocks_edge_z;
dim3 threads_edge;

dim3 dimBlock;
dim3 dimGrid ;

//function to allocate device memory and initialize data
void gpuInitialize(InitialData& id, DataArray& host, DataArray& device1, DataArray& device2)
{
	if (host.xmax % BLOCKSIZE != 0 || host.ymax % BLOCKSIZE != 0 || host.zmax % BLOCKSIZE != 0)
	{
		char buf[1024];
		sprintf(buf, "All dimensions must be multiple of %d", BLOCKSIZE);
		//Error(buf);
	}

	//Calculate kernel configurations

	blocks_laplace.x  = (int)ceil((float)host.xmax/(CELLH-2));	
	blocks_laplace.y  = (int)ceil((float)host.ymax/(CELLD-2));	
	threads_laplace.x = CELLW;
	threads_laplace.y = CELLH;
	threads_laplace.z = CELLD;

	//Kernel configuration for advection term
	blocks_adv.x  = (int)ceil((float)host.xmax/(BDIMX));
    blocks_adv.y  = (int)ceil((float)host.ymax/(BDIMY));
    threads_adv.x = BDIMX;
    threads_adv.y = BDIMY;

//	blocks_adv_x.x  = 4*host.ymax/BLOCKSIZE;
//	blocks_adv_x.y  = host.zmax;
//	threads_adv_x.x = host.xmax;
//  threads_adv_x.y = BLOCKSIZE/4;

//	blocks_adv_y.x  = 4*host.xmax/BLOCKSIZE; 
//    blocks_adv_y.y  = host.zmax;
//    threads_adv_y.x = BLOCKSIZE/4;
//    threads_adv_y.y = host.ymax;

//	blocks_adv_z.x  = (int)ceil((float)host.zmax/(BDIMZ));
//    blocks_adv_z.y  = (int)ceil((float)host.xmax/(BDIMX));
//    threads_adv_z.x = BDIMZ;
//    threads_adv_z.y = BDIMX;

	//Kernel configuration for diffusion term
	blocksDG_x.x=host.ymax;
    blocksDG_x.y=host.zmax;

    blocksDG_y.x=host.xmax;
    blocksDG_y.y=host.zmax;

    blocksDG_z.x=host.xmax;
    blocksDG_z.y=host.ymax;

    threadsDG_x.x=host.xmax/2;
    threadsDG_x.y=1;

    threadsDG_y.x=host.ymax/2;
    threadsDG_y.y=1;

    threadsDG_z.x=host.zmax/2;
    threadsDG_z.y=1;

	//Kernel configuration for transposition xyz->zxy
	grid_xyz2zxy.x=host.xmax/BLOCKSIZE;
	grid_xyz2zxy.y=host.zmax/BLOCKSIZE;
	grid_xyz2zxy.z=host.ymax;

	threads_xyz2zxy.x=BLOCKSIZE;
	threads_xyz2zxy.y=BLOCKSIZE;
	threads_xyz2zxy.z=1;

	//Kernel configuration for transposition zxy->xyz
	grid_zxy2xyz.x=host.zmax/BLOCKSIZE;
    grid_zxy2xyz.y=host.xmax/BLOCKSIZE;
    grid_zxy2xyz.z=host.ymax;

    threads_zxy2xyz.x=BLOCKSIZE;
    threads_zxy2xyz.y=BLOCKSIZE;
    threads_zxy2xyz.z=1;

	//Kernel configuration for Neumann BC
	blocks_boundary_x.x = host.ymax/BLOCKSIZE;
	blocks_boundary_x.y = host.zmax/BLOCKSIZE;

	blocks_boundary_y.x = host.xmax/BLOCKSIZE;
	blocks_boundary_y.y = host.zmax/BLOCKSIZE;

	blocks_boundary_z.x = host.xmax/BLOCKSIZE;
	blocks_boundary_z.y = host.ymax/BLOCKSIZE;

	threads_boundary.x = BLOCKSIZE;
	threads_boundary.y = BLOCKSIZE;

	blocks_edge_x.x = host.xmax/BLOCKSIZE;
	blocks_edge_y.x = host.ymax/BLOCKSIZE;
	blocks_edge_z.x = host.zmax/BLOCKSIZE;
	threads_edge.x = BLOCKSIZE;

	//Kernel conffiguration for max/min reduction
	dimBlock.x = 256;
    dimGrid.x  = 256;

	//allocate device arrays
	int size = total * sizeof(double);
	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)	
	cudaMalloc((void**)&device1.a		, size);
	cudaMalloc((void**)&device2.a		, size);
	cudaMalloc((void**)&device1.a_first , size);
	cudaMalloc((void**)&device1.a_rows      , size);
    cudaMalloc((void**)&device1.a_cols      , size);
	cudaMalloc((void**)&device1.a_ax        , size);
    cudaMalloc((void**)&device1.a_ay        , size);
    cudaMalloc((void**)&device1.a_az        , size);
    cudaMalloc((void**)&device1.a_bx        , size);
    cudaMalloc((void**)&device1.a_by        , size);
    cudaMalloc((void**)&device1.a_bz        , size);
    cudaMalloc((void**)&device1.a_cx        , size);
    cudaMalloc((void**)&device1.a_cy        , size);
    cudaMalloc((void**)&device1.a_cz        , size);
	// Copy host to device
	cudaMemcpy(device1.a            , host.a                , size              , cudaMemcpyHostToDevice);
    cudaMemcpy(device2.a            , host.a                , size              , cudaMemcpyHostToDevice);
	cudaMemcpy(device1.a_first      , host.a_first          , size              , cudaMemcpyHostToDevice);
	cudaMemcpy(device1.a_rows       , host.a_rows           , size              , cudaMemcpyHostToDevice);
    cudaMemcpy(device1.a_cols       , host.a_cols           , size              , cudaMemcpyHostToDevice);
	#endif

	//printf("GPU initialization passed.\n");
	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
	cudaMalloc((void**)&device1.b		, size);
	cudaMalloc((void**)&device2.b		, size);
	cudaMalloc((void**)&device1.b_rows      , size);
    cudaMalloc((void**)&device1.b_cols      , size);
	cudaMalloc((void**)&device1.b_ax       	, size);
    cudaMalloc((void**)&device1.b_ay       	, size);
    cudaMalloc((void**)&device1.b_az       	, size);
    cudaMalloc((void**)&device1.b_bx       	, size);
    cudaMalloc((void**)&device1.b_by       	, size);
    cudaMalloc((void**)&device1.b_bz       	, size);
    cudaMalloc((void**)&device1.b_cx       	, size);
    cudaMalloc((void**)&device1.b_cy       	, size);
    cudaMalloc((void**)&device1.b_cz       	, size);
	cudaMalloc((void**)&device1.del_b       , size);
	// Copy host to device
    cudaMemcpy(device1.b            , host.b                , size              , cudaMemcpyHostToDevice);
    cudaMemcpy(device2.b            , host.b                , size              , cudaMemcpyHostToDevice);
	cudaMemcpy(device1.b_rows       , host.b_rows           , size              , cudaMemcpyHostToDevice);
    cudaMemcpy(device1.b_cols       , host.b_cols           , size              , cudaMemcpyHostToDevice);
	cudaMemcpy(device1.del_b        , host.del_b            , size              , cudaMemcpyHostToDevice);
	#endif
	CudaCheck();
	//printf("GPU initialization passed.\n");
	#if ( MODEL == KSMD || MODEL == KSCMD )
	cudaMalloc((void**)&device1.ecm         , size);
	cudaMalloc((void**)&device2.ecm         , size);
	cudaMalloc((void**)&device1.ecm_rows    , size);
    cudaMalloc((void**)&device1.ecm_cols    , size);
	cudaMalloc((void**)&device1.ecm_ax      , size);
    cudaMalloc((void**)&device1.ecm_ay      , size);
    cudaMalloc((void**)&device1.ecm_az      , size);
    cudaMalloc((void**)&device1.ecm_bx      , size);
    cudaMalloc((void**)&device1.ecm_by      , size);
    cudaMalloc((void**)&device1.ecm_bz      , size);
    cudaMalloc((void**)&device1.ecm_cx      , size);
    cudaMalloc((void**)&device1.ecm_cy      , size);
    cudaMalloc((void**)&device1.ecm_cz      , size);
	cudaMalloc((void**)&device1.del_ecm     , size);
	//Copy host to device
	cudaMemcpy(device1.ecm          , host.ecm              , size              , cudaMemcpyHostToDevice);
    cudaMemcpy(device2.ecm          , host.ecm              , size              , cudaMemcpyHostToDevice);
    cudaMemcpy(device1.ecm_rows     , host.ecm_rows         , size              , cudaMemcpyHostToDevice);
    cudaMemcpy(device1.ecm_cols     , host.ecm_cols         , size              , cudaMemcpyHostToDevice);
    cudaMemcpy(device1.del_ecm      , host.del_ecm          , size              , cudaMemcpyHostToDevice);
	#endif
	//printf("GPU initialization passed.\n");

    cudaMalloc((void**)&device1.dt   	, sizeof(double));
    cudaMalloc((void**)&device2.dt         	, sizeof(double));
    cudaMalloc((void**)&device1.d_max_block	, 256*sizeof(double));
	cudaMemcpy(device1.dt           , host.dt               , sizeof(double)    , cudaMemcpyHostToDevice);
    cudaMemcpy(device2.dt           , host.dt               , sizeof(double)    , cudaMemcpyHostToDevice);
    cudaMemcpy(device1.d_max_block  , host.d_max_block      , 256*sizeof(double), cudaMemcpyHostToDevice);

	CudaCheck();
	//printf("GPU initialization passed.\n");

	//copy data to constant memory on device
	cudaMemcpyToSymbol(DX   , &id.dx     , sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(XMAX , &host.xmax, sizeof(int   ), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(YMAX , &host.ymax, sizeof(int   ), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(ZMAX , &host.zmax, sizeof(int   ), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NTOT , &total    , sizeof(int   ), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Zsize, &zsize    , sizeof(int   ), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Ysize, &ysize    , sizeof(int   ), 0, cudaMemcpyHostToDevice);
	CudaCheck();

	double _dx;
	_dx = 1.0/id.dx;
	cudaMemcpyToSymbol(_DX, &_dx, sizeof(double), 0, cudaMemcpyHostToDevice);
	CudaCheck();

	double _Dt;
	_Dt = 1.0/id.dt_imp;
	cudaMemcpyToSymbol(_dt  , &_Dt, sizeof(double), 0, cudaMemcpyHostToDevice);
	CudaCheck();
	printf("GPU initialization completed.\n");	
}

//compute time step for explicit method
void gpuStep_explicit(DataArray& device1, DataArray& device2, InitialData& id, double chi_dx, double chi_ecm_dx, int iter, int time_iter)
{
	//Advection - explicit Lax-Wendroff with MUSCL Flux Limiter
	#if ( MODEL == KSC || MODEL == TEST)
	kernelAdv<<<blocks_adv, threads_adv>>>(device2.a, device1.a, device1.b  , device1.del_b  , device1.dt, id.chi, chi_dx);
	kernelGrowA<<<blocks_adv, threads_adv>>>(device1.a, device2.a, device1.dt, id.s);
	//kernelAleeA<<<blocks_adv, threads_adv>>>(device1.a, device2.a, device1.a_first, device1.dt, id.s, time_iter);
	//kernelGrowB<<<blocks_adv, threads_adv>>>(device2.b, device1.b, device1.a , device1.dt, id.r);
	//swap
	//double *tempb=device2.b  ; device2.b=device1.b    ; device1.b=tempb;
	#endif

	#if ( MODEL == KSMD )
	kernelAdv<<<blocks_adv, threads_adv>>>(device2.a, device1.a, device1.ecm, device1.del_ecm, device1.dt, id.chi_ecm, chi_ecm_dx);
	kernelGrowA<<<blocks_adv, threads_adv>>>(device1.a, device2.a, device1.dt, id.s);
	//kernelAleeA<<<blocks_adv, threads_adv>>>(device1.a, device2.a, device1.a_first, device1.dt, id.s, time_iter);
	#endif

	#if ( MODEL == KSCMD )
	kernelAdv<<<blocks_adv, threads_adv>>>(device2.a, device1.a, device1.b  , device1.del_b  , device1.dt, id.chi, chi_dx);
	kernelAdv<<<blocks_adv, threads_adv>>>(device1.a, device2.a, device1.ecm, device1.del_ecm, device1.dt, id.chi_ecm, chi_ecm_dx);
    kernelGrowA<<<blocks_adv, threads_adv>>>(device2.a, device1.a, device1.dt, id.s);
	//kernelAleeA<<<blocks_adv, threads_adv>>>(device2.a, device1.a, device1.a_first, device1.dt, id.s, time_iter);
    //kernelGrowB<<<blocks_adv, threads_adv>>>(device2.b, device1.b, device1.a , device1.dt, id.r);
	//swap
    double *tempa=device2.a  ; device2.a=device1.a    ; device1.a=tempa;
    //double *tempb=device2.b  ; device2.b=device1.b    ; device1.b=tempb;
	#endif

	#if ( MODEL == FK )
	kernelGrowA<<<blocks_adv, threads_adv>>>(device2.a, device1.a, device1.dt, id.s);
	//kernelAleeA<<<blocks_adv, threads_adv>>>(device2.a, device1.a, device1.a_first, device1.dt, id.s, time_iter);
	//swap
	double *tempa=device2.a  ; device2.a=device1.a    ; device1.a=tempa;
	#endif
	

	//Update Neumann B.C.
    kernelBoundaryX<<<blocks_boundary_x, threads_boundary>>>(device1.a);
    kernelBoundaryY<<<blocks_boundary_y, threads_boundary>>>(device1.a);
    kernelBoundaryZ<<<blocks_boundary_z, threads_boundary>>>(device1.a); 
    kernelBoundaryEdgeX<<<blocks_edge_x, threads_edge>>>(device1.a);
    kernelBoundaryEdgeY<<<blocks_edge_y, threads_edge>>>(device1.a);
    kernelBoundaryEdgeZ<<<blocks_edge_z, threads_edge>>>(device1.a);

	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
    kernelBoundaryX      <<<blocks_boundary_x, threads_boundary>>>(device1.b);
    kernelBoundaryY      <<<blocks_boundary_y, threads_boundary>>>(device1.b);
    kernelBoundaryZ      <<<blocks_boundary_z, threads_boundary>>>(device1.b);
    kernelBoundaryEdgeX<<<blocks_edge_x, threads_edge>>>(device1.b);
    kernelBoundaryEdgeY<<<blocks_edge_y, threads_edge>>>(device1.b);
    kernelBoundaryEdgeZ<<<blocks_edge_z, threads_edge>>>(device1.b);
	#endif

	//Stability
	#if   ( MODEL == KSCMD )
	kernelStability<<<dimGrid, dimBlock>>>(device1.del_b, device1.del_ecm, device1.d_max_block);
	final_kernelStability<<<1, dimBlock>>>(device1.d_max_block, device1.dt, device2.dt,256, iter);//, id.da, id.db, id.dc);
	#elif ( MODEL == KSMD )
    double *p; 
    kernelStability<<<dimGrid, dimBlock>>>(p            , device1.del_ecm, device1.d_max_block);
	final_kernelStability<<<1, dimBlock>>>(device1.d_max_block, device1.dt, device2.dt,256, iter);//, id.da, 0, id.dc);
	#elif ( MODEL == KSC || MODEL == TEST )
	double *p; 
	kernelStability<<<dimGrid, dimBlock>>>(device1.del_b, p              , device1.d_max_block);
	final_kernelStability<<<1, dimBlock>>>(device1.d_max_block, device1.dt, device2.dt,256, iter);//, id.da, id.db, 0);
	#endif

	//swap
	double *tempdt=device2.dt; device2.dt=device1.dt; device1.dt=tempdt;
	CudaCheck(); //because of asynchronous execution, there will be several steps before it can return an error, if any

}


void test_CD(DataArray& device1, DataArray& device2, InitialData& id)
{
    kernelDiffusion<<<blocks_laplace, threads_laplace>>>(device1.a, device2.a, id.dt_imp, id.da);
    double *tempa=device2.a; device2.a=device1.a; device1.a=tempa;

    //Update Neumann B.C.
    kernelBoundaryX<<<blocks_boundary_x, threads_boundary>>>(device1.a);
    kernelBoundaryY<<<blocks_boundary_y, threads_boundary>>>(device1.a);
    kernelBoundaryZ<<<blocks_boundary_z, threads_boundary>>>(device1.a); //device1.del_b);
    kernelBoundaryEdgeX<<<blocks_edge_x, threads_edge>>>(device1.a);
    kernelBoundaryEdgeY<<<blocks_edge_y, threads_edge>>>(device1.a);
    kernelBoundaryEdgeZ<<<blocks_edge_z, threads_edge>>>(device1.a);
    CudaCheck();
}


//compute time step for implicit method
void gpuStep_implicit(DataArray& device1, DataArray& device2, double cfl_a, double cfl_b, double cfl_c)
{
	//Diffusion step for cell density
	fillTridiagonals_x<<<blocksDG_x,threadsDG_x>>>(device1.a_ax, device1.a_bx, device1.a_cx, device1.a, device1.a_rows, 2, cfl_a);
	fillTridiagonals_y<<<blocksDG_y,threadsDG_y>>>(device1.a_ay, device1.a_by, device1.a_cy, device1.a, device1.a_rows, device1.a_cols, 2, cfl_a);
	transpose_XYZ_to_ZXY<<<grid_xyz2zxy, threads_xyz2zxy>>>(device2.a,device1.a);
	fillTridiagonals_z_transp<<<blocksDG_z, threadsDG_z>>>(device1.a_az, device1.a_bz, device1.a_cz, device1.a_cols, device2.a, 2, cfl_a);
	transpose_ZXY_to_XYZ<<<grid_zxy2xyz, threads_zxy2xyz>>>(device1.a,device2.a);
//	getLastCudaError("cell:GPU_transpose_ZXY_to_XYZ execution failed\n");

	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
	//Diffusion step for signal density
	fillTridiagonals_x<<<blocksDG_x,threadsDG_x>>>(device1.b_ax, device1.b_bx, device1.b_cx, device1.b, device1.b_rows, 2, cfl_b);
    fillTridiagonals_y<<<blocksDG_y,threadsDG_y>>>(device1.b_ay, device1.b_by, device1.b_cy, device1.b, device1.b_rows, device1.b_cols, 2, cfl_b);
    transpose_XYZ_to_ZXY<<<grid_xyz2zxy, threads_xyz2zxy>>>(device2.b,device1.b);
    fillTridiagonals_z_transp<<<blocksDG_z, threadsDG_z>>>(device1.b_az, device1.b_bz, device1.b_cz, device1.b_cols, device2.b, 2, cfl_b);
    transpose_ZXY_to_XYZ<<<grid_zxy2xyz, threads_zxy2xyz>>>(device1.b,device2.b);
	#endif

	//Diffusion step for ECM density
	#if ( MODEL == KSMD || MODEL == KSCMD)
    fillTridiagonals_x<<<blocksDG_x,threadsDG_x>>>(device1.ecm_ax, device1.ecm_bx, device1.ecm_cx, device1.ecm, device1.ecm_rows, 2, cfl_c);
    fillTridiagonals_y<<<blocksDG_y,threadsDG_y>>>(device1.ecm_ay, device1.ecm_by, device1.ecm_cy, device1.ecm, device1.ecm_rows, device1.ecm_cols, 2, cfl_c);
    transpose_XYZ_to_ZXY<<<grid_xyz2zxy, threads_xyz2zxy>>>(device2.ecm,device1.ecm);
    fillTridiagonals_z_transp<<<blocksDG_z, threadsDG_z>>>(device1.ecm_az, device1.ecm_bz, device1.ecm_cz, device1.ecm_cols, device2.ecm, 2, cfl_c);
    transpose_ZXY_to_XYZ<<<grid_zxy2xyz, threads_zxy2xyz>>>(device1.ecm,device2.ecm);
	#endif

	//Updated Neumann B.C.
	kernelBoundaryX<<<blocks_boundary_x, threads_boundary>>>(device1.a);
    kernelBoundaryY<<<blocks_boundary_y, threads_boundary>>>(device1.a);
    kernelBoundaryZ<<<blocks_boundary_z, threads_boundary>>>(device1.a);//device1.del_b);
    kernelBoundaryEdgeX<<<blocks_edge_x, threads_edge>>>(device1.a);
    kernelBoundaryEdgeY<<<blocks_edge_y, threads_edge>>>(device1.a);
    kernelBoundaryEdgeZ<<<blocks_edge_z, threads_edge>>>(device1.a);

	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
    kernelBoundaryX<<<blocks_boundary_x, threads_boundary>>>(device1.b);
    kernelBoundaryY<<<blocks_boundary_y, threads_boundary>>>(device1.b);
    kernelBoundaryZ<<<blocks_boundary_z, threads_boundary>>>(device1.b);
    kernelBoundaryEdgeX<<<blocks_edge_x, threads_edge>>>(device1.b);
    kernelBoundaryEdgeY<<<blocks_edge_y, threads_edge>>>(device1.b);
    kernelBoundaryEdgeZ<<<blocks_edge_z, threads_edge>>>(device1.b);
	#endif

	//Update Neumann, Dirichlet BC for ECM
	#if ( MODEL == KSMD || MODEL == KSCMD)
	kernelBoundaryX_Dirichlet<<<blocks_boundary_x, threads_boundary>>>(device1.ecm);
    kernelBoundaryY_Dirichlet<<<blocks_boundary_y, threads_boundary>>>(device1.ecm);
    kernelBoundaryZ_Mix<<<blocks_boundary_z, threads_boundary>>>(device1.ecm);//device1.del_b);
    kernelBoundaryEdgeX_Mix<<<blocks_edge_x, threads_edge>>>(device1.ecm);
    kernelBoundaryEdgeY_Mix<<<blocks_edge_y, threads_edge>>>(device1.ecm);
    kernelBoundaryEdgeZ_Mix<<<blocks_edge_z, threads_edge>>>(device1.ecm);
	#endif

    //No need for swapping between device1 and device2
	CudaCheck(); //because of asynchronous execution, there will be several steps before it can return an error, if any

}


void ExportDT(DataArray& host, DataArray& device1)
{
    cudaMemcpy(host.new_dt    , device1.dt    , sizeof(double), cudaMemcpyDeviceToHost);
    CudaCheck();
}

void gpuClose(DataArray& device1, DataArray& device2)
{
	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
    cudaFree(device1.a);
    cudaFree(device2.a);
	cudaFree(device1.a_rows);
    cudaFree(device1.a_cols);
	cudaFree(device1.a_ax);
    cudaFree(device1.a_ay);
    cudaFree(device1.a_az);
    cudaFree(device1.a_bx);
    cudaFree(device1.a_by);
    cudaFree(device1.a_bz);
    cudaFree(device1.a_cx);
    cudaFree(device1.a_cy);
    cudaFree(device1.a_cz);
	#endif

	#if ( MODEL == KSC || MODEL == KSCMD )
    cudaFree(device1.b);
    cudaFree(device2.b);
	cudaFree(device1.b_rows);
    cudaFree(device1.b_cols);
	cudaFree(device1.b_ax);
    cudaFree(device1.b_ay);
    cudaFree(device1.b_az);
    cudaFree(device1.b_bx);
    cudaFree(device1.b_by);
    cudaFree(device1.b_bz);
    cudaFree(device1.b_cx);
    cudaFree(device1.b_cy);
    cudaFree(device1.b_cz);
	cudaFree(device1.del_b);
	#endif

	#if ( MODEL == KSMD || MODEL == KSCMD )
	cudaFree(device1.ecm);
    cudaFree(device2.ecm);
	cudaFree(device1.ecm_rows);
    cudaFree(device1.ecm_cols);
	cudaFree(device1.ecm_ax);
    cudaFree(device1.ecm_ay);
    cudaFree(device1.ecm_az);
    cudaFree(device1.ecm_bx);
    cudaFree(device1.ecm_by);
    cudaFree(device1.ecm_bz);
    cudaFree(device1.ecm_cx);
    cudaFree(device1.ecm_cy);
    cudaFree(device1.ecm_cz);
	#endif

	#if (MODEL == KSC || MODEL == KSMD || MODEL == KSCMD)
    cudaFree(device1.dt);
    cudaFree(device2.dt);
    cudaFree(device1.d_max_block);
	#endif

//	cudaDeviceReset();

	//sleep(5);
}

//copy data from device to host and write it to a binary file
void ExportCheckpoint(const char* name_a, const char* name_b, const char* name_c, DataArray& host, DataArray& device1, float l, int printout)
{
    cudaMemcpy(host.a_intm   , device1.a    , total*sizeof(double), cudaMemcpyDeviceToHost);
	#if ( MODEL == KSC  || MODEL == KSCMD || MODEL == TEST )
    cudaMemcpy(host.b_intm   , device1.b    , total*sizeof(double), cudaMemcpyDeviceToHost);
	#endif
	#if ( MODEL == KSMD || MODEL == KSCMD )
	cudaMemcpy(host.ecm_intm , device1.ecm  , total*sizeof(double), cudaMemcpyDeviceToHost);
	#endif
    CudaCheck();

#ifdef SAVE_DATA
    //create a file
    char filename_a[4096];
	#if   ( MODEL == FK   )
	const char* name_m = "fk";
	#elif ( MODEL == TEST )
	const char* name_m = "test";
	#elif ( MODEL == KSC  )
	const char* name_m = "ksc";
	#elif ( MODEL == KSMD )
	const char* name_m = "ksmd";
	#elif ( MODEL == KSCMD )
	const char* name_m = "kscmd";
	#endif

	sprintf(filename_a, "%s_%s_%f.vtk", name_a, name_m, l);
    VTKsave(filename_a, host.xmax, host.ymax, host.zmax, host.a_intm);
	
    #if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
	char filename_b[4096];
    sprintf(filename_b, "%s_%s_%f.vtk", name_b, name_m, l);
    VTKsave(filename_b, host.xmax, host.ymax, host.zmax, host.b_intm); 
	#endif

	#if ( MODEL == KSMD || MODEL == KSCMD)
	char filename_c[4096];
//	FILE *fc;
	sprintf(filename_c, "%s_%f.vtk", name_c, l);
    VTKsave(filename_c, host.xmax, host.ymax, host.zmax, host.c_intm);
/*
	fc = fopen(filename_c, "wb");
	if (fc==0) { printf("  error creating %s\n", filename_c); return; }
	fwrite(host.ecm_intm, total * sizeof(double), 1, fc);
	fclose(fc);
*/
	#endif

	double suma=0.;
	for(int i=0;i<total;i++){
		suma += host.a_intm[i];
	}
	printf("summed density = %lf",suma);

#endif

/*
	//create a file
	const char* name_as="a_slice_ks";
        const char* name_bs="b_slice_ks";
        char filename_as[4096];
        char filename_bs[4096];
        FILE *fas;
        FILE *fbs;
        sprintf(filename_as, "%s_%d.txt", name_as, l);
        sprintf(filename_bs, "%s_%d.txt", name_bs, l);
        fas = fopen(filename_as, "w");
        fbs = fopen(filename_bs, "w");
        if (fas==0) { printf("  error creating %s\n", filename_as); return; }
        if (fbs==0) { printf("  error creating %s\n", filename_bs); return; }


	//Export a slice
        int cs=0;
        for(int k=0; k<host.zmax; k++){
                for(int j=0; j<host.ymax; j++){
                        for(int i=0; i<host.xmax; i++){

                                int id = i + host.xmax*(j + host.ymax*k);
                                if(i==host.xmax/2 && j==host.ymax/2)
                                {
                                        //host.a_slice[cs] = host.a_intm[id];
                                        //host.b_slice[cs] = host.b_intm[id];
					fprintf(fas,"%d	%1.20f\n", cs, host.a_intm[id]);
					fprintf(fbs,"%d	%1.20f\n", cs, host.b_intm[id]);
                                        cs++;
                                }
                        }
                }
        }

        fclose(fas);
        fclose(fbs);
*/
	if (printout) {
		double* max_host_a = max_element(host.a_intm, host.a_intm+int(zsize*host.zmax));
        double* min_host_a = min_element(host.a_intm, host.a_intm+int(zsize*host.zmax));
        printf("max_a = %1.2e\nmin_a = %1.2e\n", *max_host_a, *min_host_a);

        for (int k=2.0*host.zmax/5.0-1; k<2.0*host.zmax/5.0+2; k++) {
        for (int j=2.0*host.ymax/5.0-1; j<2.0*host.ymax/5.0+4; j++) {
        for (int i=2.0*host.xmax/5.0-1; i<2.0*host.xmax/5.0+4; i++) {
            int id = i + host.xmax*(j + k*host.ymax);
            printf(" %1.2e ", host.a_intm[id]);
        }
            printf("\n");
        }
            printf("\n");
        }

		#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
		double* max_host_b = max_element(host.b_intm, host.b_intm+int(zsize*host.zmax));
        double* min_host_b = min_element(host.b_intm, host.b_intm+int(zsize*host.zmax));
        printf("max_b = %1.2e\nmin_b = %1.2e\n", *max_host_b, *min_host_b);

        for (int k=2.0*host.zmax/5.0-1; k<2.0*host.zmax/5.0+2; k++) {
        for (int j=2.0*host.ymax/5.0-1; j<2.0*host.ymax/5.0+4; j++) {
        for (int i=2.0*host.xmax/5.0-1; i<2.0*host.xmax/5.0+4; i++) {
            int id = i + host.xmax*(j + k*host.ymax);
            printf(" %1.2e ", host.b_intm[id]);
        }
            printf("\n");
        }
            printf("\n");
        }

		#endif

		#if ( MODEL == KSMD || MODEL == KSCMD)
		double* max_host_ecm = max_element(host.ecm_intm, host.ecm_intm+int(zsize*host.zmax));
        double* min_host_ecm = min_element(host.ecm_intm, host.ecm_intm+int(zsize*host.zmax));
        printf("max_ecm = %1.2e\nmin_ecm = %1.2e\n", *max_host_ecm, *min_host_ecm);

		for (int k=2.0*host.zmax/5.0-1; k<2.0*host.zmax/5.0+2; k++) {
        for (int j=2.0*host.ymax/5.0-1; j<2.0*host.ymax/5.0+4; j++) {
        for (int i=2.0*host.xmax/5.0-1; i<2.0*host.xmax/5.0+4; i++) {
            int id = i + host.xmax*(j + k*host.ymax);
            printf(" %1.2e ", host.ecm_intm[id]);
        }
            printf("\n");
        }
            printf("\n");
        }
		#endif
    }
}

/*
void gpuKa(InitialData& id)
{
	cudaMemcpyToSymbol(Ka, &id.ka, sizeof(double), 0, cudaMemcpyHostToDevice);
}
*/
//free device memory

