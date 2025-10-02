#pragma once
#include "vec3.h"
#include <SDL2/SDL.h>

void drawSphere3D(SDL_Renderer *renderer, Vec3 center, float radius, Vec3 lightPos, Vec3 camPos);
void drawLightSource(SDL_Renderer *renderer, Vec3 lightPos, Vec3 camPos);
