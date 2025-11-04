#pragma once
#include "vec3.h"

struct Camera {
    Vec3 position;
    Vec3 lightPos;
    float angleX, angleY;

    Camera();
    Vec3 rotatePoint(const Vec3 &point) const;
};
