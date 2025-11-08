#include "vec3.h"

constexpr float REFRESH_RATE_30FPS = 0.033f;
constexpr float REFRESH_RATE_60FPS = 0.016f;
constexpr float REFRESH_RATE_120FPS = 0.008f;

constexpr double NEWTON_GRAVITATIONAL_CONSTANT = 6.67430e-11;
constexpr double DISTANCE_SCALE = 1e9;
constexpr double TIME_SCALE = 86400.0;
constexpr double G_SCALED =
    NEWTON_GRAVITATIONAL_CONSTANT * TIME_SCALE * TIME_SCALE / (DISTANCE_SCALE * DISTANCE_SCALE * DISTANCE_SCALE);

constexpr double EARTH_MASS = 5.972e24;
constexpr double SUN_MASS = 1.989e30;

constexpr double SMOOTHING_LENGTH = 20.0;
constexpr double REST_DENSITY = 1.0;
constexpr double GAS_CONSTANT = 500.0;
constexpr double MU = 0.1;
constexpr double POLY6 = 315.0 / (64.0 * M_PI * 1.953125e15);
constexpr double SPIKYGRAD = -45.0 / (M_PI * 1.5625e10);
constexpr double VISCLAPLACIAN = 45.0 / (M_PI * 1.5625e10);

static const Vec3 BLUE(0.0f, 0.0f, 255.0f);
static const Vec3 YELLOW(255.0f,255.0f, 0.0f);
static const Vec3 RED(255.0f, 0.0f, 0.0f);
static const Vec3 WHITE(255.0f, 255.0f, 255.0f);
static const Vec3 ORANGE(255.0f, 165.0f, 0.0f);