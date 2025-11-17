#pragma once
#include <cmath>

struct Vec3 {
    float x, y, z;

    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    Vec3 operator-(const Vec3 &other) const { return Vec3(x - other.x, y - other.y, z - other.z); }

    Vec3 operator+(const Vec3 &other) const { return Vec3(x + other.x, y + other.y, z + other.z); }

    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }

    Vec3 &operator+=(const Vec3 &other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec3 &operator-=(const Vec3 &other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    float dot(const Vec3 &other) const { return x * other.x + y * other.y + z * other.z; }

    float lengthSquared() const { return x * x + y * y + z * z; }

    float length() const { return std::sqrt(x * x + y * y + z * z); }

    Vec3 normalized() const {
        float len = length();
        return len > 0 ? Vec3(x / len, y / len, z / len) : Vec3(0, 0, 0);
    }

    Vec3 cross(const Vec3 &other) const {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
    
    Ray(Vec3 o, Vec3 d) : origin(o), direction(d.normalized()) {}
    
    Vec3 at(float t) const {
        return origin + direction * t;
    }
};
