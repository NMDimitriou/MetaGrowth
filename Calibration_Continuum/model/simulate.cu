#include "declarations.cuh"

/* Initialize simulation parameters and process
 * Nikolaos Dimitriou, McGill, 2021
*/

double simulate(DataArray& host, DataArray& device1, DataArray& device2, ExpData& expdat, InitialData& id)
{
	//define parameters
    id.dx              = 2.5f/(ysize -1);
	double sigma_data2 = id.param*id.param;
	id.SSE       = 0;
	#if ( STUDY == SA )
	id.tot_num = 0.0;
	id.bot_num = 0.0;
	#endif

	// Implicit time step based Da, Db
	id.cfl       = 1.0;
	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
	double dt_a  = id.cfl*id.dx*id.dx/id.da;
	id.dt_imp    = dt_a;
	#endif
	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
	double dt_b  = id.cfl*id.dx*id.dx/id.db;
	id.dt_imp    = min(dt_a,dt_b);
	#endif
	#if ( MODEL == KSMD || MODEL == KSCMD )
	double dt_c  = id.cfl*id.dx*id.dx/id.dc;
	id.dt_imp    = min(id.dt_imp,dt_c);
	#endif

	id.dt_imp    = min(id.dt_imp,0.1); 
    id.itmax     = 14.0/id.dt_imp;

	#if ( MODEL == FK || MODEL == KSC || MODEL == KSMD || MODEL == KSCMD || MODEL == TEST)
    double cfl_a = id.da*id.dt_imp/(id.dx*id.dx);
	#endif
	#if ( MODEL == KSC || MODEL == KSCMD || MODEL == TEST)
    double cfl_b = id.db*id.dt_imp/(id.dx*id.dx);
	double chi_b_dx   = id.chi/id.dx;
	#endif
	#if   ( MODEL == KSMD || MODEL == KSCMD )
	double cfl_c = id.dc*id.dt_imp/(id.dx*id.dx);
	double chi_ecm_dx = id.chi_ecm/id.dx;
	#endif

	//PRINT MESSAGE	
	print_message(id,sigma_data2);

	//WRITE NRMSE
	FILE* fnrmse;
	#ifdef NRMSE
	fnrmse = save_nrmse(host);
	#endif


	//INITIALIZE
    hostInitialize(host);
    #if ( MODEL == FK )
    *host.dt     = 0.25*id.dt_imp;
    *host.new_dt = 0.25*id.dt_imp;
    #endif
    gpuInitialize(id, host, device1, device2);


	int exportInterval[6] = {int(2.0f/id.dt_imp),int(5.0f/id.dt_imp),int(7.0f/id.dt_imp),
	            			 int(9.0f/id.dt_imp),int(12.0f/id.dt_imp),int(14.0f/id.dt_imp)} ;
    int c                 = 0    ;
	double step_dt        = 0.   ; //explicit time-step
    int ct = 0; //ct: counter of function calls
	double se; //sum of errors
	
	//events for time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //COMPUTE
	printf("Starting loop\n");
	printf("Implicit time-step = %.10f\n",id.dt_imp);
    printf("Explicit time-step = %.10f (initial)\n",*host.dt);
	cudaEventRecord(start, 0);
    for (int i = 1; i <= id.itmax; i++)
    {
	    //Strang splitting
		//EXPLICIT STEP
		#if ( MODEL == FK || MODEL == KSCMD || MODEL == KSMD || MODEL == KSC || MODEL == TEST)
        while(step_dt - 0.5*id.dt_imp < -1.0e-12)
        {
			ct ++;
			#if   ( MODEL == KSCMD )
            gpuStep_explicit(device1, device2, id, chi_b_dx, chi_ecm_dx, ct, i);
			#elif ( MODEL == KSMD  )
			gpuStep_explicit(device1, device2, id, 0.0     , chi_ecm_dx, ct, i);
			#elif ( MODEL == KSC   || MODEL == TEST)
			gpuStep_explicit(device1, device2, id, chi_b_dx, 0.0       , ct, i);
			#elif ( MODEL == FK    )
			gpuStep_explicit(device1, device2, id, 0.0     , 0.0       , ct, i);
			#endif
            ExportDT(host,device1);
            step_dt += *host.new_dt;
              
			// What if dt is too small?
			if(*host.new_dt < *host.dt/1000000.0)
            {
                printf("Solution probably will not converge\n");
                printf("or takes a long time to converge.\n");
                printf("Exiting\n");
				#if ( STUDY == CALIBRATION)
				status(-1);
				#endif

				#if ( STUDY == SA)
				id.tot_num=0; id.bot_num=0;
                goto STOP1;
                #else
				gpuClose(device1, device2);
                hostClose(host);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                exit(-1);
                #endif
            }
			//printf("Host.new_dt = %.15f\n", *host.new_dt);
        }
		step_dt = 0.;
		#endif

		//IMPLICIT STEP
		#if   ( MODEL == KSCMD )
        gpuStep_implicit(device1, device2, cfl_a, cfl_b, cfl_c);
		#elif ( MODEL == KSMD  )
		gpuStep_implicit(device1, device2, cfl_a, 0.0  , cfl_c);
		#elif ( MODEL == KSC  || MODEL == TEST )
		gpuStep_implicit(device1, device2, cfl_a, cfl_b, 0.0  );
		//test_CD(device1, device2, id);
		#elif ( MODEL == FK    )
		gpuStep_implicit(device1, device2, cfl_a, 0.0  , 0.0  );
		#endif
        ct ++;

		//EXPLICIT STEP
		#if ( MODEL == FK || MODEL == KSCMD || MODEL == KSMD || MODEL == KSC || MODEL == TEST)
        while(step_dt - 0.5*id.dt_imp < -1.0e-12)
        {
			ct ++;
			#if   ( MODEL == KSCMD )
            gpuStep_explicit(device1, device2, id, chi_b_dx, chi_ecm_dx, ct, i);
            #elif ( MODEL == KSMD  )
            gpuStep_explicit(device1, device2, id, 0.0     , chi_ecm_dx, ct, i);
            #elif ( MODEL == KSC   || MODEL == TEST)
            gpuStep_explicit(device1, device2, id, chi_b_dx, 0.0       , ct, i);
            #elif ( MODEL == FK    )
            gpuStep_explicit(device1, device2, id, 0.0     , 0.0       , ct, i);
            #endif
            ExportDT(host,device1);
            step_dt += *host.new_dt;
                        
			// What if dt is too small?
			if(*host.new_dt < *host.dt/1000000.0)
            {
                printf("Solution probably will not converge\n");
                printf("or takes a long time to converge.\n");
                printf("Exiting\n");
				#if ( STUDY == CALIBRATION)
				status(-1);
				#endif
				
				#if ( STUDY == SA)
				id.tot_num=0; id.bot_num=0;
                goto STOP1;
                #else
                gpuClose(device1, device2);
                hostClose(host);
                //hostCloseExp(expdat);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                exit(-1);
                #endif

            }
			//printf("Host.new_dt = %.15f\n", *host.new_dt);
        }
		#endif
		step_dt = 0.;

        if ((int)i == exportInterval[c])
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop); //gives in milliseconds
            elapsed /= 1000.0f; //convert to seconds
            float mpoints = (float)(exportInterval[c]/(*host.new_dt)) * host.xmax * host.ymax * host.zmax / 1.0e6f;//reportInterval , exportInterval[c]                  
            printf("t_sim = %.2f: %f Mpoints/s \n", i*id.dt_imp, mpoints / elapsed);
			printf("function calls = %d\n",ct);
	        ExportCheckpoint("a_double","b_double","ecm_double",host,device1,ceil(i*id.dt_imp),0);

			#if (STUDY == CALIBRATION)
			printf("Calculating SSE/RMSE/NRMSE.\n");
			printf("###############################\n");
			calcResiduals(c,se,id,host,expdat,fnrmse);
			#elif ( STUDY == SA)
			printf("Calculating responses.\n");
            printf("###############################\n");
			calcResponse(id,host);
			#endif
			printf("###############################\n\n");
            c++;
			cudaEventRecord(start, 0);
        }
        if (i % int(0.1/id.dt_imp) == 0)
        {
            ExportCheckpoint("a_double","b_double","ecm_double",host,device1,i*id.dt_imp,0);
        }
		#if (STUDY == CALIBRATION)
		if(id.SSE != id.SSE)
		{
            printf("SSE is Nan or Inf! Exiting loop. \n");
			status(0);
            break;
        }
		#elif ( STUDY == SA)
		if(id.bot_num != id.bot_num || id.tot_num != id.tot_num)
        {               
            printf("Response is Nan or Inf! Exiting loop. \n");
            break;
        }
		#endif


        }

#if ( STUDY == SA)
STOP1:	
#endif

    //FINISH
    gpuClose(device1, device2);
	hostClose(host);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	#if (STUDY == CALIBRATION)
	if(id.SSE==id.SSE) { status(1); }else{ id.SSE=1e7;/*negative if used instead of loglike*/ }
	printf("Total SSE = %.15f\n", id.SSE);
	double loglike = -0.5*total*log(2*M_PI) - 0.5*total*log(sigma_data2) - 0.5*id.SSE/sigma_data2; //
	//double loglike = -id.SSE;
	printf("Log-Likelighood = %.15f\n", loglike);
	printf(" \n");
    return loglike;	
	#else
	return 0;
	#endif

}//end
