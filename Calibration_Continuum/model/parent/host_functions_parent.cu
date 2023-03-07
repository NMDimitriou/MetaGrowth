#include "declarations_parent.cuh"

//allocate memory on host
void hostInitializeParent(DataArray& host, char* hostname)
{
	int size      = total * sizeof(double);

	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
	host.segid_a	  = shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	host.segid_a_rows = shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
    host.segid_a_cols = shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	host.a            = (double *)shmat(host.segid_a,NULL,0);
	host.a_rows       = (double *)shmat(host.segid_a_rows,NULL,0);
	host.a_cols       = (double *)shmat(host.segid_a_cols,NULL,0);
	if (host.a           == NULL) Error("Host.a memory allocation failed\n");
	if (host.a_rows      == NULL) Error("Host.a_rows memory allocation failed\n");
	if (host.a_cols      == NULL) Error("Host.a_cols memory allocation failed\n");
	#endif

	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
	host.segid_b	    =shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	host.segid_b_rows   =shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
    host.segid_b_cols   =shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	host.segid_del_b    =shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	host.b           = (double *)shmat(host.segid_b,NULL,0);
    host.b_rows      = (double *)shmat(host.segid_b_rows,NULL,0);
    host.b_cols      = (double *)shmat(host.segid_b_cols,NULL,0);
    host.del_b       = (double *)shmat(host.segid_del_b ,NULL,0);
	if (host.b           == NULL) Error("Host.b memory allocation failed\n");
    if (host.b_rows      == NULL) Error("Host.b_rows memory allocation failed\n");
    if (host.b_cols      == NULL) Error("Host.b_cols memory allocation failed\n");
    if (host.del_b       == NULL) Error("Host.del_b memory allocation failed\n");
	#endif

	#if ( MODEL == KSMD || MODEL == KSCMD)
    host.segid_ecm          =shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
    host.segid_ecm_rows     =shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
    host.segid_ecm_cols     =shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
    host.segid_del_ecm      =shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	host.ecm         = (double *)shmat(host.segid_ecm,NULL,0);
    host.ecm_rows    = (double *)shmat(host.segid_ecm_rows,NULL,0);
    host.ecm_cols    = (double *)shmat(host.segid_ecm_cols,NULL,0);
    host.del_ecm     = (double *)shmat(host.segid_del_ecm ,NULL,0);
    if (host.ecm         == NULL) Error("Host.ecm memory allocation failed\n");
    if (host.ecm_rows    == NULL) Error("Host.ecm_rows memory allocation failed\n");
    if (host.ecm_cols    == NULL) Error("Host.ecm_cols memory allocation failed\n");
    if (host.del_ecm     == NULL) Error("Host.del_ecm memory allocation failed\n");
    #endif

    host.segid_d_max_block	=shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
    host.d_max_block = (double *)shmat(host.segid_d_max_block,NULL,0);
	if (host.d_max_block == NULL) Error("Host.d_max_block memory allocation failed\n");

	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD)
	char filename[4096];
    sprintf(filename, "%s/new_dens_%s_D0.raw", host.path, host.dat);
	FILE *f = fopen(filename,"rb");
	if (f!=NULL)
	{
	   	printf("Loading Initial Conditions ... \r\n");
	   	fread(host.a,sizeof(double),total,f);
	}else
	{
	   	printf("Initial Conditions for Cells not found. \r\n");
		printf("Make sure that the file is found in ../../IC\n");
		printf("and the filename that follows the pattern: corr_dens_[name of dataset e.g. AE]_D0.bin\n");
	   	exit(-1);
	}
	fclose(f);
	#elif ( MODEL == TEST)
    for (int k=0; k<host.zmax; k++) {
    for (int j=0; j<host.ymax; j++) {
    for (int i=0; i<host.xmax; i++) {
        int ind = i + j*host.xmax + k*host.xmax*host.ymax;
        if(i>2.0*host.xmax/5.0 && i<3.0*host.xmax/5.0 && j>2.0*host.ymax/5.0 && j<3.0*host.ymax/5.0 && k>3.0*host.zmax/5.0 && k<4.0*host.zmax/5.0)
        {       
            host.a[ind]=0.1;
        }
        else
        {
            host.a[ind]=0.0;
        }
    }
    }
    }
	printf("This is a test.\n");
	#endif
	memcpy(host.a_rows,host.a,size);
	memcpy(host.a_cols,host.a,size);

	#if ( MODEL == KSMD || MODEL == KSCMD)
	char fnameECM[4096];
    sprintf(fnameECM, "%s/ecm_%s_D0.bin", host.path,host.dat);
    FILE *ff = fopen(fnameECM,"rb");
    if (ff!=NULL)
    {
        printf("Loading Initial Conditions ... \r\n");
        fread(host.ecm,sizeof(double),total,ff);
    }else
    {
        printf("Initial Conditions for ECM not found. \r\n");
        exit(-1);
    }
	fclose(ff);
    memcpy(host.ecm_rows,host.ecm,size);
    memcpy(host.ecm_cols,host.ecm,size);
	#endif

	#if ( MODEL == KSC || MODEL == KSCMD)
    double count  = 0.0;
	int c         = 0;
	for (int j=0; j<host.zmax;j++)
    {
        for (int i=0; i<zsize; i++)
        {
            //host.b[c] = (1.0-0.005681818*count);
            //host.b[c] = exp(-count/20.0)*host.a[c]/(*max_a);
			//host.b[c] = host.a[c];

			if(host.a[c]>1.0e-12){
				#if   ( IC == EXP )
				host.b[c] = exp(-count/50.0);
				#elif ( IC == LIN )
				host.b[c] = (1.0-0.005681818*count);
				#elif ( IC == ID  )
				host.b[c] = host.a[c];
				#elif ( IC == NA )
				host.b[c] = 0.0;
				#endif
			}
			else{
				host.b[c] = 0.0;
			}
            c ++;
        }
        count ++;
    }
	memcpy(host.b_rows,host.b,size);
	memcpy(host.b_cols,host.b,size);
	#elif ( MODEL == TEST )
	double count  = 0.0;
    int c         = 0;
    for (int j=0; j<host.zmax;j++)
    {
        for (int i=0; i<zsize; i++)
        {
            host.b[c] = (1.0-0.005681818*count);
            c ++;
        }
        count ++;
    }
    memcpy(host.b_rows,host.b,size);
    memcpy(host.b_cols,host.b,size);
	#endif

	for(int i=0; i<256; i++)
    {
        host.d_max_block[i]=100.0;
    }


	char fnameIC[4096];
	const char* name = "../inIC";
    sprintf(fnameIC, "%s_%s.txt", name, hostname);

	FILE* fin;
    fin = fopen(fnameIC, "w");
	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
    fprintf(fin, "%d\n", host.segid_a);
	fprintf(fin, "%d\n", host.segid_a_rows);
	fprintf(fin, "%d\n", host.segid_a_cols);
	#endif
	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
    fprintf(fin, "%d\n", host.segid_b);
	fprintf(fin, "%d\n", host.segid_b_rows);
    fprintf(fin, "%d\n", host.segid_b_cols);
	fprintf(fin, "%d\n", host.segid_del_b);
	#endif
	#if ( MODEL == KSMD || MODEL == KSCMD)
    fprintf(fin, "%d\n", host.segid_ecm);
    fprintf(fin, "%d\n", host.segid_ecm_rows);
    fprintf(fin, "%d\n", host.segid_ecm_cols);
    fprintf(fin, "%d\n", host.segid_del_ecm);
    #endif

    fprintf(fin, "%d\n", host.segid_d_max_block);
    fclose(fin);	
}

#if ( STUDY == CALIBRATION )
void hostExpInitParent(DataArray& host, ExpData& expdat, char* hostname)
{
    int size     = total * sizeof(double);

	expdat.segid_dzero = shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	expdat.segid_dtwo  = shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	expdat.segid_dfive = shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	expdat.segid_dsvn  = shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	expdat.segid_dnine = shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	expdat.segid_dtwlv = shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);
	expdat.segid_dfrtn = shmget(IPC_PRIVATE,size,S_IRUSR|S_IWUSR);

	expdat.dzero = (double *)shmat(expdat.segid_dzero,NULL,0);
	expdat.dtwo  = (double *)shmat(expdat.segid_dtwo ,NULL,0);
	expdat.dfive = (double *)shmat(expdat.segid_dfive,NULL,0);
	expdat.dsvn  = (double *)shmat(expdat.segid_dsvn ,NULL,0);
	expdat.dnine = (double *)shmat(expdat.segid_dnine,NULL,0);
	expdat.dtwlv = (double *)shmat(expdat.segid_dtwlv,NULL,0);
	expdat.dfrtn = (double *)shmat(expdat.segid_dfrtn,NULL,0);

    if (expdat.dzero == NULL) Error("Host memory allocation failed\n");
    if (expdat.dtwo  == NULL) Error("Host memory allocation failed\n");
    if (expdat.dfive == NULL) Error("Host memory allocation failed\n");
    if (expdat.dsvn  == NULL) Error("Host memory allocation failed\n");
    if (expdat.dnine == NULL) Error("Host memory allocation failed\n");
    if (expdat.dtwlv == NULL) Error("Host memory allocation failed\n");
    if (expdat.dfrtn == NULL) Error("Host memory allocation failed\n");

	char filename0[4096], filename2[4096], filename5[4096], filename7[4096], filename9[4096], 
	     filename12[4096], filename14[4096];
    sprintf(filename0,  "%s/new_dens_%s_D0.raw", host.path, host.dat);
	sprintf(filename2,  "%s/new_dens_%s_D2.raw", host.path, host.dat);
	sprintf(filename5,  "%s/new_dens_%s_D5.raw", host.path, host.dat);
	sprintf(filename7,  "%s/new_dens_%s_D7.raw", host.path, host.dat);
	sprintf(filename9,  "%s/new_dens_%s_D9.raw", host.path, host.dat);
	sprintf(filename12, "%s/new_dens_%s_D12.raw", host.path, host.dat);
	sprintf(filename14, "%s/new_dens_%s_D14.raw", host.path, host.dat);
	
    FILE *fzer = fopen(filename0 ,"rb");
    FILE *ftwo = fopen(filename2 ,"rb");
    FILE *ffiv = fopen(filename5 ,"rb");
    FILE *fsev = fopen(filename7 ,"rb");
    FILE *fnin = fopen(filename9 ,"rb");
    FILE *ftwl = fopen(filename12,"rb");
    FILE *ffrt = fopen(filename14,"rb");


    printf("Loading experimental data...\r\n");
    if (fzer!=NULL)
    {
//      printf("Day 0...\r\n");
        fread(expdat.dzero,sizeof(double),total,fzer);
        fclose(fzer);

    }
    else
    {
        printf("Day zero not found.\n");
		printf("Make sure that the file is found in ../../IC\n");
        printf("and the filename that follows the pattern: corr_dens_[name of dataset e.g. AE]_D0.bin\n");
        exit(-1);
    }
    if (ftwo!=NULL)
    {
//              printf("Day 2...\r\n");
        fread(expdat.dtwo,sizeof(double),total,ftwo);
        fclose(ftwo);
	}
    else
    {
        printf("Day two not found.\n");
		printf("Make sure that the file is found in ../../IC\n");
        printf("and the filename that follows the pattern: corr_dens_[name of dataset e.g. AE]_D2.bin\n");
        exit(-1);
    }
    if (ffiv!=NULL)
    {
//              printf("Day 5...\r\n");
        fread(expdat.dfive,sizeof(double),total,ffiv);
        fclose(ffiv);
    }
    else
    {
        printf("Day five not found.\n");
		printf("Make sure that the file is found in ../../IC\n");
        printf("and the filename that follows the pattern: corr_dens_[name of dataset e.g. AE]_D5.bin\n");
        exit(-1);
    }
    if (fsev!=NULL)
    {
//              printf("Day 7...\r\n");
        fread(expdat.dsvn,sizeof(double),total,fsev);
        fclose(fsev);
    }
    else
    {
        printf("Day seven not found.\n");
		printf("Make sure that the file is found in ../../IC\n");
        printf("and the filename that follows the pattern: corr_dens_[name of dataset e.g. AE]_D7.bin\n");
        exit(-1);
    }
    if (fnin!=NULL)
    {
//              printf("Day 9...\r\n");
        fread(expdat.dnine,sizeof(double),total,fnin);
        fclose(fnin);
    }
    else
    {
        printf("Day nine not found.\n");
		printf("Make sure that the file is found in ../../IC\n");
        printf("and the filename that follows the pattern: corr_dens_[name of dataset e.g. AE]_D9.bin\n");
        exit(-1);
    }
    if (ftwl!=NULL)
    {
//              printf("Day 12...\r\n");
        fread(expdat.dtwlv,sizeof(double),total,ftwl);
        fclose(ftwl);
    }
    else
	{
        printf("Day twelve not found.\n");
		printf("Make sure that the file is found in ../../IC\n");
        printf("and the filename that follows the pattern: corr_dens_[name of dataset e.g. AE]_D12.bin\n");
        exit(-1);
    }
    if (ffrt!=NULL)
    {
//              printf("Day 14...\r\n");
        fread(expdat.dfrtn,sizeof(double),total,ffrt);
        fclose(ffrt);
    }
    else
    {
        printf("Day fourteen not found.\n");
		printf("Make sure that the file is found in ../../IC\n");
        printf("and the filename that follows the pattern: corr_dens_[name of dataset e.g. AE]_D14.bin\n");
        exit(-1);
    }

	char filename[4096];
        const char* name = "../inExp";
        sprintf(filename, "%s_%s.txt", name, hostname);

	FILE* fin;
    fin = fopen(filename, "w");
    fprintf(fin, "%d\n", expdat.segid_dzero);
    fprintf(fin, "%d\n", expdat.segid_dtwo);
    fprintf(fin, "%d\n", expdat.segid_dfive);
    fprintf(fin, "%d\n", expdat.segid_dsvn);
    fprintf(fin, "%d\n", expdat.segid_dnine);
    fprintf(fin, "%d\n", expdat.segid_dtwlv);
    fprintf(fin, "%d\n", expdat.segid_dfrtn);
    fclose(fin);

    printf("Loading complete. \r\n");
}
#endif

//free memory on host
void hostClose(DataArray& host)
{
	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
    free(host.a    );
    free(host.a_rows);
    free(host.a_cols);

	shmdt(host.a);
	shmdt(host.a_rows);
	shmdt(host.a_cols);
	shmctl(host.segid_a,IPC_RMID,NULL);
	shmctl(host.segid_a_rows,IPC_RMID,NULL);
	shmctl(host.segid_a_cols,IPC_RMID,NULL);
	#endif
	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
    free(host.b    );
    free(host.b_rows);
    free(host.b_cols);
    free(host.del_b);

	shmdt(host.b);
	shmdt(host.b_rows);
	shmdt(host.b_cols);
	shmdt(host.del_b);
	shmctl(host.segid_b,IPC_RMID,NULL);
    shmctl(host.segid_b_rows,IPC_RMID,NULL);
    shmctl(host.segid_b_cols,IPC_RMID,NULL);
    shmctl(host.segid_del_b,IPC_RMID,NULL);
	#endif

	#if ( MODEL == KSMD || MODEL == KSCMD)
    free(host.ecm);
    free(host.ecm_rows);
    free(host.ecm_cols);
    free(host.del_ecm);

	shmdt(host.ecm);
	shmdt(host.ecm_rows);
	shmdt(host.ecm_cols);
	shmdt(host.del_ecm);
	shmctl(host.segid_ecm,IPC_RMID,NULL);
    shmctl(host.segid_ecm_rows,IPC_RMID,NULL);
    shmctl(host.segid_ecm_cols,IPC_RMID,NULL);
    shmctl(host.segid_del_ecm,IPC_RMID,NULL);
	#endif

    free(host.d_max_block);
	shmdt(host.d_max_block);
    shmctl(host.segid_d_max_block,IPC_RMID,NULL);


//	free(host.a_slice);
//	free(host.b_slice);
}

#if ( STUDY == CALIBRATION )
void hostCloseExp(ExpData& expdat)
{
    free(expdat.dzero);
    free(expdat.dtwo) ;
    free(expdat.dfive);
    free(expdat.dsvn) ;
    free(expdat.dnine);
    free(expdat.dtwlv);
    free(expdat.dfrtn);

	shmdt(expdat.dzero);
    shmdt(expdat.dtwo);
    shmdt(expdat.dfive);
    shmdt(expdat.dsvn);
    shmdt(expdat.dnine);
    shmdt(expdat.dtwlv);
    shmdt(expdat.dfrtn);

    shmctl(expdat.segid_dzero,IPC_RMID,NULL);
    shmctl(expdat.segid_dtwo,IPC_RMID,NULL);
    shmctl(expdat.segid_dfive,IPC_RMID,NULL);
    shmctl(expdat.segid_dsvn,IPC_RMID,NULL);
    shmctl(expdat.segid_dnine,IPC_RMID,NULL);
    shmctl(expdat.segid_dtwlv,IPC_RMID,NULL);
    shmctl(expdat.segid_dfrtn,IPC_RMID,NULL);

}
#endif
