#include <gpgpu.h>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <crt/math_functions.h>


/** @brief Permert d'obtenir les informations GPU
 */
void GetGPGPUInfo() {
	cudaDeviceProp cuda_propeties;
	cudaGetDeviceProperties(&cuda_propeties, 0);
	std::cout << "maxThreadsPerBlock: " << cuda_propeties.maxThreadsPerBlock << std::endl;
}

/** @brief Retourne la partie decimale
   @param float x : nombre décimale
   @return float : partie decimale */
__device__ float fracf(float x) {
	//prend la nombre x moins sa partie entière
	return x - floorf(x); 
}

/** @brief Obtient un nombre pseudo aléatoire. Les deux paramètres doivent avoir des seeds qui changent (*time)
   @param float x et y : seeds
   @return float : nombre pseudo aléatoire */
__device__ float random(float x, float y)
{
	float t = 12.9898f * x + 78.233f*y;
	return abs(fracf(t * sin(t)));
}

/** @brief Calcule la distance euclidienne entre deux animaux
   @param Animaux a1, Animaux a2 : Animal 1 et Animal 2
   @return float : distance entre les deux animaux*/
__device__ float  distance(Animaux a1, Animaux a2) {
	float deltaU = a1.u - a2.u;
	float deltaV = a1.v - a2.v;
	return sqrt(deltaU * deltaU + deltaV * deltaV);
}

/** @brief Dessine le fond de la fenêtre
   @param cudaSurfaceObject_t surface : structure de l'image
   @param int width et height : dimensions de l'image
   @return void*/
__global__ void Draw_Background(cudaSurfaceObject_t surface, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 color = make_float4(0.6f, 0.9f, 0.05f, 1.0f);
	surf2Dwrite(color, surface, x * sizeof(float4), y);
}

/*  fonction avant optimisation
__global__ void Draw_Crcl(cudaSurfaceObject_t surface, Animaux * lapin) {
	
	float4 color;
	int ind =  threadIdx.x;
	//printf("Boucle lapin %d : %d \n", ind, lapin[ind].alive);

	if (lapin[ind].alive == true)
	{
		float x = lapin[ind].u - lapin[ind].radius;
		float y = lapin[ind].v - lapin[ind].radius;
		//printf("draw lapin (%d) :  %f \n", ind, lapin[ind].u);
		for (int i = x; i < (lapin[ind].u + 2 * lapin[ind].radius); i++)
		{
			//printf("boucle de [X %d] [Ind %d] \n ", i,ind);
			for (int j = y; j < (lapin[ind].v + 2 * lapin[ind].radius); j++)
			{
				if (hypotf(lapin[ind].u - float(i), lapin[ind].v - float(j)) < lapin[ind].radius)
				{
					if (i > 0 && i < kWidth && j > 0 && j < kHeight) {
						color = lapin[ind].color;
						surf2Dwrite(color, surface, i * sizeof(float4), j);
					}
				}
			}
		}
	}
}*/

/** @brief Dessine le cercle représentant les animaux
   @param cudaSurfaceObject_t surface : structure de l'image
   @param Animaux * animal : Tableau des animaux
   @return void*/
__global__ void Draw_Crcl(cudaSurfaceObject_t surface, Animaux * animal) {
   float4 color;
   int ind =  threadIdx.x;
   if (animal[ind].alive) {
      float x = animal[ind].u - animal[ind].radius;
      float y = animal[ind].v - animal[ind].radius;
      float x2 = animal[ind].u + animal[ind].radius;
      float y2 = animal[ind].v + animal[ind].radius;

      float r2 = animal[ind].radius * animal[ind].radius;
      color = animal[ind].color;

      for (int i = max(0, (int)x); i <= min(kWidth-1, (int)x2); i++) {
         for (int j = max(0, (int)y); j <= min(kHeight-1, (int)y2); j++) {
            float dx = animal[ind].u - (float)i;
            float dy = animal[ind].v - (float)j;
            if (dx*dx + dy*dy < r2) {
               surf2Dwrite(color, surface, i * sizeof(float4), j);
            }
         }
      }
   }
}

/** @brief Si un renard touche un lapin, alors il le mange
   @param Animaux* Renard : Renard
   @param Animaux* Lapin : Lapin
   @return void*/
__global__ void Manger(Animaux* Renard , Animaux* Lapin) {
	int ind = threadIdx.x;
	bool aManger = false;

	if (Renard[ind].alive == true){
		for (int i = 0; i < NB_LAPINS; i++) {
			if (Lapin[i].alive == true) {
				float d = distance(Lapin[i], Renard[ind]);
				//si le lapin et le renard se touchent (la somme des deux radius) 
				if (d <= Lapin[i].radius + Renard[ind].radius) {
					Lapin[i].alive = false;
					aManger = true;
					Renard[ind].diet = 0;
					break;
				}
			}
		}
		//Si un Renard ne mange pas pendant trop longtemps, il meurt
		if (!aManger) {
			if (Renard[ind].diet == DUREE_VIE_RENARD) {
				Renard[ind].alive = false;
			} else {
				Renard[ind].diet++;
			}
		}
	}
}

/** @brief Si deux lapins se rencontrent, alors ils donnent vie à un nouveau lapin
   @param Animaux * Lapin : Lapin en cours de traitement
   @return void*/
__global__ void Reproduire(Animaux* Lapin) {
  int ind = threadIdx.x;
  int copain = -1;
  int mort = -1;

  if (Lapin[ind].alive == true) {
    for (int i = 0; i < NB_LAPINS; i++) {
      if ( copain == -1 && Lapin[i].alive && i != ind) {
        float d = distance(Lapin[i], Lapin[ind]);
        if (d <= Lapin[i].radius + Lapin[ind].radius) {
          copain = i;
        }
      } else if (Lapin[i].alive == false) {
        mort = i;
      }

      if (copain != -1 && mort != -1) {
        break;
      }
    }

    if (copain != -1 && mort != -1) {
      Lapin[mort].alive = true;
      Lapin[mort].u = (Lapin[ind].u + Lapin[copain].u) / 2;
      Lapin[mort].v = (Lapin[ind].v + Lapin[copain].v) / 2;
    }
  }
}


/** @brief Si un renard est proche d'un lapin, alors il le poursuit
   @param Animaux* Renard : Chasseur
   @param Animaux* Lapin : Chassé
   @param float time : time pour avoir des seeds random pour les déplacements
   @return void*/
__global__ void TrouverVoisinProche(Animaux* Renard,Animaux* Lapin , float time) {

	int ind = threadIdx.x;
	float distanceMin = 1024;
	int indiceVoisin = -1;
	if (Renard[ind].alive == true) {
		for (int i = 0; i < NB_LAPINS; i++) {
			if (Lapin[i].alive == true) {
				float d = distance(Lapin[i], Renard[ind]);
				if (d < distanceMin && d < FOX_R_DETECT) {
					distanceMin = d;
					indiceVoisin = i;
				}
			}
		}

		if (indiceVoisin != -1){
			float dx = Lapin[indiceVoisin].u - Renard[ind].u;
			float dy = Lapin[indiceVoisin].v - Renard[ind].v;
			float angle = atan2(dy, dx);
			Renard[ind].angle = angle;
		} else {
			Renard[ind].angle += (((random(Renard[ind].v * time, Renard[ind].u) - .5) * 2) * 3.14 / 5);
		}
	}
}

/** @brief Crée un trajectoire pseudo random
   @param Animaux* Animaux : tableau des Animaux en cours de traitement
   @param float time : time pour avoir des seeds random pour les déplacements
   @return void*/
__global__ void Deplacment(Animaux* Animaux, float time) {
	int ind = threadIdx.x;
	
	if (Animaux[ind].alive == true)	{
		//lapin[ind].norm = ((random(lapin[ind].u * time, lapin[ind].v) - .5));
		if (Animaux[ind].type == LAPIN) {
			Animaux[ind].angle += (((random(Animaux[ind].v * time, Animaux[ind].u) - .5) * 2) * 3.14 / 5);
		}
		Animaux[ind].u += cos(Animaux[ind].angle) * (Animaux[ind].norm);
		(Animaux[ind].u > 1024) ? Animaux[ind].u = 0 : ((Animaux[ind].u < 0) ? Animaux[ind].u = 1024 : Animaux[ind].u);
		//lapin[ind].u = float(int(lapin[ind].u) % 1024) + fracf(lapin[ind].u);
		Animaux[ind].v += sin(Animaux[ind].angle) * (Animaux[ind].norm);
		(Animaux[ind].v > 1024) ? Animaux[ind].v = 0 : ((Animaux[ind].v < 0) ? Animaux[ind].v = 1024 : Animaux[ind].v);
		//lapin[ind].v = float(int(lapin[ind].v) % 1024) + fracf(lapin[ind].v);
	}
}


__global__ void kernel_copy(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	float4 color = make_float4(1.f, 0.f, 1.f, 1.0f);
	surf2Dread(&color, surface_in, x * sizeof(float4), y);
	surf2Dwrite(color, surface_out, x * sizeof(float4), y);
}


/** @brief Mets les fonctions du GPU en action
   @param cudaSurfaceObject_t surface 
   @param Animaux * v_lapin, v_renard
   @param int32_t width, height
   @param float time
   @return void*/
void DrawUVs(cudaSurfaceObject_t surface,Animaux * v_lapin, Animaux* v_renard, int32_t width, int32_t height, float time) {

	//affichage du fond 
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	Draw_Background <<< blocks, threads >>> (surface, width, height);
	
	// copie de donner CPU vers memoir GPU
	Animaux* fox_d;
	cudaMalloc(&fox_d, sizeof(Animaux) * NB_FOX);
	cudaMemcpy(fox_d, v_renard, sizeof(Animaux) * NB_FOX, cudaMemcpyHostToDevice);

	Animaux* rabbits_d;
	cudaMalloc(&rabbits_d, sizeof(Animaux) * NB_LAPINS);
	cudaMemcpy(rabbits_d, v_lapin, sizeof(Animaux) * NB_LAPINS, cudaMemcpyHostToDevice);

	//traitement sur les renards
	dim3 th_f(NB_FOX, 1);
	dim3 bl_f(1, 1);
	Manger <<< bl_f, th_f >> > (fox_d, rabbits_d);
	TrouverVoisinProche <<< bl_f, th_f >>> (fox_d, rabbits_d,time);
	Deplacment <<< bl_f, th_f >>> (fox_d, time);
	Draw_Crcl <<< bl_f, th_f >>> (surface, fox_d);
	
	// traitement sur les lapins
	dim3 th_l(NB_LAPINS, 1);
	dim3 bl_l(1, 1);
	Deplacment <<< bl_l, th_l >>> (rabbits_d, time);
	if (int(time * 100) % FREQENCE_REPRODUCTION == 0)
		Reproduire <<< bl_l, th_l >>> (rabbits_d);

	Draw_Crcl <<< bl_l, th_l >>> (surface, rabbits_d);


	// liberation donner 
	cudaMemcpy(v_lapin, rabbits_d, sizeof(Animaux) * NB_LAPINS, cudaMemcpyDeviceToHost);
	cudaFree(rabbits_d);

	cudaMemcpy(v_renard, fox_d, sizeof(Animaux) * NB_FOX, cudaMemcpyDeviceToHost);
	cudaFree(fox_d);

}

void CopyTo(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out, int32_t width, int32_t height) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	kernel_copy <<< blocks, threads >>> (surface_in, surface_out);
}