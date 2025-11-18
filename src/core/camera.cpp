#include "camera.h"
#include <cmath>

Camera::Camera() {
    position = Vec3(400, 100, -100);
    lookAt = Vec3(400, 300, 200);
    up = Vec3(0, 1, 0);
    lightPos = Vec3(200, -200, 300);
    fov = 60.0f;
    aspectRatio = 800.0f / 600.0f;
    angleX = -M_PI / 2.5f;
    angleY = 0.0f;
    updateCameraVectors();
}

void Camera::updateCameraVectors() {
    float theta = fov * M_PI / 180.0f;
    float h = tan(theta / 2.0f);
    float focusDistance = (position - lookAt).length();
    
    viewportHeight = 2.0f * h * focusDistance;
    viewportWidth = aspectRatio * viewportHeight;
    
    w = (position - lookAt).normalized();
    u = up.cross(w).normalized();
    v = w.cross(u);
    
    horizontal = u * viewportWidth;
    vertical = v * viewportHeight;
    lowerLeftCorner = position - horizontal * 0.5f - vertical * 0.5f - w * focusDistance;
}

Ray Camera::getRay(float s, float t) const {
    Vec3 pixelPos = lowerLeftCorner + horizontal * s + vertical * t;
    return Ray(position, pixelPos - position);
}

Vec3 Camera::rotatePoint(const Vec3 &point) const {
    float cosY = std::cos(angleY);
    float sinY = std::sin(angleY);
    Vec3 rotY(point.x * cosY - point.z * sinY, point.y, point.x * sinY + point.z * cosY);

    float cosX = std::cos(angleX);
    float sinX = std::sin(angleX);
    return Vec3(rotY.x, rotY.y * cosX - rotY.z * sinX, rotY.y * sinX + rotY.z * cosX);
}
