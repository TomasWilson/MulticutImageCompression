#pragma once
#include <vector>
#include <typeindex>
#include <functional>
#include <string>
#include <unordered_map>
#include <any>

#define REPORT_DIAGNOSTICS

#ifdef REPORT_DIAGNOSTICS

template<typename T>
std::ostream& operator<<(std::ostream& os, std::optional<T> const& opt)
{
    return opt ? os << opt.value() : os;
}

// This utility class should allow reporting arbitrary diagnostic data,
// for instance the entropy of the huffman coder, without having to modify the function signature
// the functionality can be disabled by unsetting the REPORT_DIAGNOSTICS flag, so that it incurs no overhead.
// A callback is identified by its parameter type and a string "key".
class Diagnostics {

    using Callback = std::function<void(const void*)>;

    std::unordered_map<std::type_index, std::unordered_map<std::string, Callback>> callbacks;
    std::unordered_map<std::type_index, std::unordered_map<std::string, std::vector<std::any>>> stored_data;

    Diagnostics() = default;

public:

    static Diagnostics& instance() {
        static Diagnostics instance;
        return instance;
    }

    template <typename T>
    void message(const std::string& key, const T& data, bool store=true) {
        auto ti = std::type_index(typeid(T));
        stored_data[ti][key].push_back(data);
        if(callbacks.find(ti) == callbacks.end()) return;
        if(callbacks.at(ti).find(key) == callbacks.at(ti).end()) return;
        callbacks.at(ti).at(key)(&data);
    }

    template<typename T>
    void register_callback(const std::string& key, std::function<void(const T&)> callback_fn) {
        auto ti = std::type_index(typeid(T));
        callbacks[ti][key] = [callback_fn](const void* data) {
            callback_fn(*static_cast<const T*>(data));
        };
    }

    template<typename T>
    std::optional<T> get(const std::string& key) { // get the last stored element for the key
        auto ti = std::type_index(typeid(T));
        if(stored_data.find(ti) == stored_data.end()) return {};
        if(stored_data.at(ti).find(key) == stored_data.at(ti).end()) return {};
        const auto& all = stored_data.at(ti).at(key);
        if(all.size() == 0) return {};
        return std::make_optional(std::any_cast<T>(all.at(all.size() - 1)));
    }

    template<typename T>
    std::vector<T> get_all(const std::string& key) {
        auto ti = std::type_index(typeid(T));
        if(stored_data.find(ti) == stored_data.end()) return {};
        if(stored_data.at(ti).find(key) == stored_data.at(ti).end()) return {};
        const auto& all = stored_data.at(ti).at(key);
        std::vector<T> res;
        res.reserve(all.size());
        for (const auto& d : all) {
            res.push_back(std::any_cast<T>(d));
        }
        return res;
    }

};

#endif

#ifdef REPORT_DIAGNOSTICS
    #define DIAGNOSTICS_REGISTER_CALLBACK(key, type, callback) \
        Diagnostics::instance().register_callback<type>(key, callback)
    #define DIAGNOSTICS_MESSAGE(key, data) \
        Diagnostics::instance().message(key, data)
    #define DIAGNOSTICS_GET(key, type) \
        Diagnostics::instance().get<type>(key)
    #define DIAGNOSTICS_GETALL(key, type) \
        Diagnostics::instance().get_all<type>(key)
#else
    #define DIAGNOSTICS_REGISTER_CALLBACK(key, type, callback) \
        do {} while (0)
    #define DIAGNOSTICS_MESSAGE(key, data) \
        do {} while (0)
    #define DIAGNOSTICS_GET(key, type) \
        do {} while (0)
    #define DIAGNOSTICS_GETALL(key, type) \
        do {} while (0)
#endif