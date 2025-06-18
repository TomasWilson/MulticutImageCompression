#include "timing.h"

std::unordered_map<int, std::chrono::system_clock::time_point> times;

void tic(int marker) {
    times[marker] = std::chrono::system_clock::now();
}

int toc(const std::string& msg, int marker) {
    auto now2 = std::chrono::system_clock::now();

    if(times.find(marker) == times.end()) {
        std::cerr << "WARNING: toc() marker was not found. Computed time is erronous." << std::endl;
    }

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now2 - times[marker]);

    if(msg.empty()) {
        std::cout << "Elapsed since last tic(): " << elapsed_ms.count() << "ms" << std::endl;
    }
    else {
        std::cout << msg << " " << elapsed_ms.count() << "ms" << std::endl;
    }

    return elapsed_ms.count();
}

int toctic(const std::string& msg, int marker) {
    int res = toc(msg, marker);
    tic(marker);
    return res;
}