#pragma once
#include <SDL2/SDL.h>
#include "vec3.h"

void drawSphere3D(SDL_Renderer* renderer, Vec3 center, float radius, Vec3 lightPos, Vec3 camPos);
void drawLightSource(SDL_Renderer* renderer, Vec3 lightPos, Vec3 camPos);
