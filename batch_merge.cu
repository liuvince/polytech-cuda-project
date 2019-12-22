#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define X 0
#define Y 1 

bool isSumOfLengthEqualtoD(int *, int *, int, int);
bool isPowerOfTwo(int);

__global__ void mergeSmallBatch_k(int * A, int * sizeA, int *B, int *sizeB, int *M, const int d, int N){

	// Question 4 : Indices importantes
	const int tidx = threadIdx.x % d;				// Numéro de la diagonal dans le tableau numéro Qt
	const int Qt = (threadIdx.x - tidx) / d;			// Numéro du tableau par rapport au tableau shared
	const int gbx = Qt + blockIdx.x * (blockDim.x / d);		// Numéro du tableau par rapport au tableau global

	const int sizeAi = sizeA[gbx];				// Taille du tableau considéré
	const int sizeBi = sizeB[gbx];				// Taille du tableau considéré

	__shared__ int Atemp[1024];				// Tableau partagé par les threads d'un bloc
	__shared__ int Btemp[1024];				// Tableau partagé par les threads d'un bloc

	Atemp[Qt * d + tidx] = A[gbx * d + tidx];		// Remplissage du tableau
	Btemp[Qt * d + tidx] = B[gbx * d + tidx];		// Remplissage du tableau

	__syncthreads();					// Attente de la synchronisation de tous les threads du bloc
//	printf("blockId.x = %d | threadIdx. x = %d | tidx = %d | Qt = %d | gbx = %d\n", blockIdx.x, threadIdx.x, tidx, Qt, gbx);

	if (gbx * d + tidx >= N * d){
		return;
	}
	// ======================== //
	//        Merge path        //
	// ======================== //
	int K[2];
	int P[2];

	if (tidx > sizeAi) {
		K[X] = tidx - sizeAi;
		K[Y] = sizeAi;
		P[X] = sizeAi;
		P[Y] = tidx - sizeAi;
	}
	else {
		K[X] = 0;
		K[Y] = tidx;
		P[X] = tidx;
		P[Y] = 0;
	}

	while (1) {
		int offset = (abs(K[Y] - P[Y]))/2;
		int Q[2] = {K[X] + offset, K[Y] - offset};

		if (Q[Y] >= 0 && Q[X] <= sizeBi && (Q[Y] == sizeAi || Q[X] == 0 || Atemp[Qt*d + Q[Y]] > Btemp[Qt*d + Q[X]-1])) {
			if (Q[X] == sizeBi || Q[Y] == 0 || Atemp[Qt*d + Q[Y]-1] <= Btemp[Qt*d + Q[X]]) {
				if (Q[Y] < sizeAi && (Q[X] == sizeBi || Atemp[Qt*d + Q[Y]] <= Btemp[Qt*d + Q[X]]) ) {
						M[gbx * d + tidx] = Atemp[Qt*d + Q[Y]];
				}
				else {
						M[gbx * d + tidx] = Btemp[Qt*d + Q[X]];
				}
				// printf("%d\n", M[gbx*d + tidx]);
				break ;
			}
			else {
				K[X] = Q[X] + 1;
				K[Y] = Q[Y] - 1;
			}
		}
		else {
			P[X] = Q[X] - 1;
			P[Y] = Q[Y] + 1 ;
		}
	}
}

int main() {

	// Graine aléatoire
	srand(0);

	// GPU Timer instructions
	float TimerV;
	cudaEvent_t start, stop;

	// ==================== //
	//      Parameters      //
	// ==================== //
	const int d = 32;
	int N = 100000;

	int threadsPerBlock = 1024;
	// ===================== //

	int numBlocks = (threadsPerBlock - 1 + N * d) / threadsPerBlock;
	// Allocation de la mémoire
	int * aHost = (int *) malloc(N*d * sizeof(int));
	int * bHost = (int *) malloc(N*d * sizeof(int));
	int * mHost = (int *) malloc(N*d * sizeof(int));

	int * sizeAHost = (int*) malloc(N * sizeof(int));
	int * sizeBHost = (int*) malloc(N * sizeof(int));

	// Remmplissage des tableaux Ai et Bi
	for (int i = 0; i < N; i++){

		// Taille aléatoire du tableau A[i]
		int alea = rand() % d;

		sizeAHost[i] = alea;
		sizeBHost[i] = d - sizeAHost[i];

		// Remplissage des tableaux avec des valeurs croissantes car le tab doit etre trié
		for (int j = 0; j < sizeAHost[i]; j ++){
			aHost[i*d+j] = 2*j;
		}
		for (int j = 0; j < sizeBHost[i]; j ++){
			bHost[i*d+j] = 2*j + 1;
		}
	}

	// Test
	{
		assert( isPowerOfTwo( d ) );
		assert( isPowerOfTwo( threadsPerBlock ) );
		assert( threadsPerBlock <= 1024 );
		assert( d <= 1024 );
		assert( threadsPerBlock % d == 0 );
		assert( isSumOfLengthEqualtoD(sizeAHost, sizeBHost, N, d) );
	}
	

	// Allouer la mémoire globale dans le GPU
	int * aDevice, * bDevice, * mDevice ;
	cudaMalloc( (void**) &aDevice, N*d * sizeof(int) );
	cudaMalloc( (void**) &bDevice, N*d * sizeof(int) );
	cudaMalloc( (void**) &mDevice, N*d * sizeof(int) );
	int * sizeADevice, * sizeBDevice;
	cudaMalloc( (void**) &sizeADevice, N * sizeof(int) );
	cudaMalloc( (void**) &sizeBDevice, N * sizeof(int) );

	// Copier les tableaux vers le GPU
	cudaMemcpy( aDevice, aHost, N*d * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( bDevice, bHost, N*d * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( sizeADevice, sizeAHost, N * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( sizeBDevice, sizeBHost, N * sizeof(int), cudaMemcpyHostToDevice );

	// GPU Timer instructions
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Lancer le kernel pour fusionner les tableaux 2 à 2
	mergeSmallBatch_k<<<numBlocks, threadsPerBlock>>>( aDevice, sizeADevice, bDevice, sizeBDevice, mDevice, d, N);

	// GPU timer instructions
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&TimerV, start, stop);
	printf("%d, %d, %f\n", N, d, TimerV);

	// Copier les tableaux du device vers host
	cudaMemcpy( mHost, mDevice, N*d * sizeof(int), cudaMemcpyDeviceToHost );

	
	// Show result for array (Ai, Bi)
	{
		int i = N-1;
		assert(i < N);

		for (int j = 0; j < d; j++){
			printf("Mhost[%d][%d] = %d\n", i, j, mHost[i*d+j]);
		}
		printf("============================\n");
		printf("Ci-dessus est le tableau M numero i=%d sur les N=%d.\n", i, N);
	        printf("C'est un tableau de taille %d, on a fusioné A et B.\n", d);
		if (sizeAHost[i] != 0)
			printf("A est le tableau de %d nombres PAIRS allant 0 à %d.\n", sizeAHost[i], aHost[d*i+sizeAHost[i]-1]);
		if (sizeBHost[i] != 0)
			printf("B est le tableau de %d nombres IMPAIRS allant 1 à %d.\n", sizeBHost[i], bHost[d*i+sizeBHost[i]-1]);
	}
	

	// Liberer la mémoire
	free(aHost);
	free(bHost);
	free(mHost);
	free(sizeAHost);
	free(sizeBHost);
	cudaFree(aDevice);
	cudaFree(bDevice);
	cudaFree(mDevice);
	cudaFree(sizeADevice);
	cudaFree(sizeBDevice);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}


bool isSumOfLengthEqualtoD(int *A, int *B, int N, int d){
	for (int i = 0; i < N; i ++){
		if (A[i] + B[i] != d){
			return false;
		}
	}
	return true;
}

// https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
bool isPowerOfTwo(int x)
{
	    return (x != 0) && ((x & (x - 1)) == 0);
}

