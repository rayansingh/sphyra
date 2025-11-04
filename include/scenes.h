#pragma once
#include <functional>
#include <string>
#include <vector>

class Scene;

struct SceneConfig {
    std::string name;
    std::string description;
    std::function<void(Scene&)> setup;
};

namespace Scenes {
    void accretionDisk(Scene& scene);

    extern std::vector<SceneConfig> all;
}
