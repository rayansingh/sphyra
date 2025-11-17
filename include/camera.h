#pragma once
#include "vec3.h"

struct Camera {
    Vec3 position;
    Vec3 lookAt;
    Vec3 up;
    Vec3 lightPos;
    float fov;
    float aspectRatio;
    
    Vec3 u, v, w;
    float viewportHeight;
    float viewportWidth;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 lowerLeftCorner;
    
    float angleX, angleY;

    Camera();
    void updateCameraVectors();
    Ray getRay(float s, float t) const;
    Vec3 rotatePoint(const Vec3 &point) const;
};
