#pragma once

#include <vector>
#include <thrust/device_vector.h>



#define RABBIT_COLOR make_float4(1.0f, 1.0f, 1.0f, 1.0f)
#define FOX_COLOR make_float4(1.0f, 0.5f, 0.0f, 1.0f)

#define FREQENCE_REPRODUCTION 20
#define DUREE_VIE_RENARD 1000

#define NB_LAPINS 500
#define NB_FOX 15

#define FOX_R_DETECT 120

constexpr int32_t kWidth = 1024;
constexpr int32_t kHeight = 1024;


enum Espece{
	LAPIN,RENARD
};


/**
    @struct Animaux
    @brief Contient des informations sur un animal dans une simulation.
    @param float u : X position of the animal.
    @param float v : Y position of the animal.
    @param float radius :  Size of the animal.
    @param float norm : Magnitude of the animal's velocity.
    @param float angle : Direction of the animal's movement.
    @param bool alive : Boolean value indicating whether the animal is alive or not.
    @param Espece type : Enumeration representing the species of the animal.
    @param float4 color :  Float4 value representing the color of the animal.
    @param int diet : time depuis dernier repas (pour fox) 
*/
struct Animaux {
	float u;
	float v;
	float radius;
	float norm;
	float angle;
	bool alive;
	Espece type;
	float4 color;
	int diet;
};





void GetGPGPUInfo();
void DrawUVs(cudaSurfaceObject_t surface, Animaux* v_lapin, Animaux* v_renard, int32_t width, int32_t height, float time);
void CopyTo(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out, int32_t width, int32_t height);