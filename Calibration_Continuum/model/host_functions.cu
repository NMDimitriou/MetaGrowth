#include "declarations.cuh"

/* Intialize parameters stored in global memory
 * Nikolaos Dimitriou, McGill, 2021
*/

//allocate memory on host
void hostInitialize(DataArray& host)
{
	int size      = total * sizeof(double);
	double dt_min = 1e-08;


	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
	host.a_intm      = (double*) malloc(size);
	#endif
	#if   ( MODEL == KSC  || MODEL == KSCMD || MODEL == TEST)
	host.b_intm      = (double*) malloc(size);
	#endif
	#if   ( MODEL == KSMD || MODEL == KSCMD )
	host.ecm_intm    = (double*) malloc(size);
	#endif

    cudaMallocHost((void**)&host.dt    , sizeof(double));
    cudaMallocHost((void**)&host.new_dt, sizeof(double));

	char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, HOST_NAME_MAX + 1);
    //printf("hostname: %s\n", hostname);
    //printf("\n");

	char filename[4096];
    const char* name = "inIC";
    sprintf(filename, "%s_%s.txt", name, hostname);

	printf("Importing Initial Conditions from shared memory\n");
    printf("from file %s\n", filename);

	#if   ( MODEL == KSCMD )
	int numpar = 12;
	#elif ( MODEL == KSC  || MODEL == TEST )
	int numpar = 8;
	#elif ( MODEL == KSMD  )
	int numpar = 7;
	#elif ( MODEL == FK    )
	int numpar = 3;
	#endif

	int ff[numpar];
  	FILE *f;
  	f = fopen(filename, "r");
  	for(int i=0;i<numpar;i++)
  	{
        fscanf(f, "%d",&ff[i]);
  	}
  	fclose(f);
	
	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
	int id_a 	   = ff[0];
	int id_a_rows      = ff[1];
	int id_a_cols      = ff[2];
	#endif
	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
	int id_b 	   = ff[3];
	int id_b_rows 	   = ff[4];
	int id_b_cols 	   = ff[5];
	int id_del_b  	   = ff[6];
	#endif
	#if   ( MODEL == KSMD || MODEL == KSCMD)
    int id_ecm         = ff[7];
    int id_ecm_rows    = ff[8];
    int id_ecm_cols    = ff[9];
    int id_del_ecm     = ff[10];
    #endif

	int id_d_max_block = ff[numpar-1];

	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
	host.a 		 	 = (double *)shmat(id_a,NULL,0);
	host.a_first     = (double *)shmat(id_a,NULL,0);
	host.a_rows      = (double *)shmat(id_a_rows,NULL,0);
	host.a_cols      = (double *)shmat(id_a_cols,NULL,0);
	if (host.a_intm      == NULL) Error("Host.a_intm memory allocation failed\n");
	if (host.a           == NULL) Error("Host.a memory allocation failed\n");
	if (host.a_rows      == NULL) Error("Host.a_rows memory allocation failed\n");
	if (host.a_cols      == NULL) Error("Host.a_cols memory allocation failed\n");
	#endif
	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
	host.b 		 = (double *)shmat(id_b,NULL,0);
    host.b_rows 	 = (double *)shmat(id_b_rows,NULL,0);
    host.b_cols 	 = (double *)shmat(id_b_cols,NULL,0);
	host.del_b  	 = (double *)shmat(id_del_b,NULL,0);
	if (host.b_intm      == NULL) Error("Host.b_intm memory allocation failed\n");
    if (host.b           == NULL) Error("Host.b memory allocation failed\n");
    if (host.b_rows      == NULL) Error("Host.b_rows memory allocation failed\n");
    if (host.b_cols      == NULL) Error("Host.b_cols memory allocation failed\n");
    if (host.del_b       == NULL) Error("Host.del_b memory allocation failed\n");
	#endif
	#if   ( MODEL == KSMD || MODEL == KSCMD)
    host.ecm         = (double *)shmat(id_ecm,NULL,0);
    host.ecm_rows    = (double *)shmat(id_ecm_rows,NULL,0);
    host.ecm_cols    = (double *)shmat(id_ecm_cols,NULL,0);
    host.del_ecm     = (double *)shmat(id_del_ecm,NULL,0);
	if (host.ecm         == NULL) Error("Host.ecm memory allocation failed\n");
    if (host.ecm_rows    == NULL) Error("Host.ecm_rows memory allocation failed\n");
    if (host.ecm_cols    == NULL) Error("Host.ecm_cols memory allocation failed\n");
    if (host.del_ecm     == NULL) Error("Host.del_ecm memory allocation failed\n");
    if (host.ecm_intm    == NULL) Error("Host.ecm_intm memory allocation failed\n");
    #endif

	host.d_max_block = (double *)shmat(id_d_max_block,NULL,0);
	if (host.d_max_block == NULL) Error("Host.d_max_block memory allocation failed\n");

    *host.dt = dt_min; //a "small" initial dt
	printf("Host initialization completed.\n");
	
}

#if (STUDY == CALIBRATION)
void hostExpInit(ExpData& expdat)
{
	
	char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, HOST_NAME_MAX + 1);
    printf("hostname: %s\n", hostname);
	printf("\n");

    char filename[4096];
    const char* name = "inExp";
    sprintf(filename, "%s_%s.txt", name, hostname);

	printf("Importing Experimental data from shared memory\n");
	printf("from file %s\n", filename);

	int ff[7];
    FILE *f;
    f = fopen(filename, "r");
    for(int i=0;i<7;i++)
    {
        fscanf(f, "%d",&ff[i]);
    }
    fclose(f);

	int id_zero	= ff[0];
    int id_two	= ff[1];
    int id_five	= ff[2];
    int id_svn	= ff[3];
    int id_nine	= ff[4];
    int id_twlv	= ff[5];
    int id_frtn	= ff[6];

    expdat.dzero = (double *)shmat(id_zero,NULL,0);
    expdat.dtwo  = (double *)shmat(id_two,NULL,0);
    expdat.dfive = (double *)shmat(id_five,NULL,0);
    expdat.dsvn  = (double *)shmat(id_svn,NULL,0);
    expdat.dnine = (double *)shmat(id_nine,NULL,0);
    expdat.dtwlv = (double *)shmat(id_twlv,NULL,0);
    expdat.dfrtn = (double *)shmat(id_frtn,NULL,0);

    if (expdat.dzero == NULL) Error("Host memory allocation failed\n");
    if (expdat.dtwo  == NULL) Error("Host memory allocation failed\n");
    if (expdat.dfive == NULL) Error("Host memory allocation failed\n");
    if (expdat.dsvn  == NULL) Error("Host memory allocation failed\n");
    if (expdat.dnine == NULL) Error("Host memory allocation failed\n");
    if (expdat.dtwlv == NULL) Error("Host memory allocation failed\n");
    if (expdat.dfrtn == NULL) Error("Host memory allocation failed\n");

	printf("Completed.\n");
    printf(" \n");

}
#endif

//free memory on host
void hostClose(DataArray& host)
{

	#if   ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
	free(host.a_intm);
	#endif
	#if   ( MODEL == KSC  || MODEL == KSCMD || MODEL == TEST)
	free(host.b_intm);
	#endif
	#if   ( MODEL == KSMD || MODEL == KSCMD )
	free(host.ecm_intm);
	#endif
/*
	free(host.a);
	free(host.b);
	free(host.a_rows);
	free(host.b_rows);
	free(host.a_cols);
    free(host.b_cols);
	free(host.del_b);
	free(host.d_max_block);
*/
//	cudaFreeHost(host.dt);
//    cudaFreeHost(host.new_dt);

//	free(host.a_slice);
//	free(host.b_slice);
}

void hostCloseExp(ExpData& expdat)
{
/*
	free(expdat.dzero);
	free(expdat.dtwo);
	free(expdat.dfive);
	free(expdat.dsvn);
	free(expdat.dnine);
	free(expdat.dtwlv);
	free(expdat.dfrtn);
*/
}

