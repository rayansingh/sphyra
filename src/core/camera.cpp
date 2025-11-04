#include "camera.h"
#include <cmath>

Camera::Camera() : position(400, 300, 0), lightPos(200, -200, 300), angleX(0), angleY(0) {}

Vec3 Camera::rotatePoint(const Vec3 &point) const {
    float cosY = std::cos(angleY);
    float sinY = std::sin(angleY);
    Vec3 rotY(point.x * cosY - point.z * sinY, point.y, point.x * sinY + point.z * cosY);

    float cosX = std::cos(angleX);
    float sinX = std::sin(angleX);
    return Vec3(rotY.x, rotY.y * cosX - rotY.z * sinX, rotY.y * sinX + rotY.z * cosX);
}
