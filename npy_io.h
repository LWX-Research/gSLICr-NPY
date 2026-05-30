// npy_io.h — Minimal NPY reader/writer for gSLICr.
// No dependencies beyond C++11 and <fstream>.
// Supports: uint8 (H,W,C) images, int32 (H,W) label maps.
//
// NPY format: magic \x93NUMPY + version(2B) + header_len(4B) + dict + raw data
// https://numpy.org/devdocs/reference/generated/numpy.lib.format.html

#pragma once

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

inline std::string build_header(const std::string& descr, bool fortran_order,
                                const std::vector<size_t>& shape) {
    std::ostringstream ss;
    ss << "{'descr': '" << descr << "', 'fortran_order': ";
    ss << (fortran_order ? "True" : "False");
    ss << ", 'shape': (";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i) ss << ", ";
        ss << shape[i];
    }
    // pad so header ends on 16-byte boundary (including \n)
    ss << ")}";
    std::string h = ss.str();
    size_t target = ((h.size() + 1) + 15) & ~15u;
    while (h.size() + 1 < target) h.push_back(' ');
    h.push_back('\n');
    return h;
}

inline void save(const std::string& path, const int* data,
                 const std::vector<size_t>& shape) {
    std::ofstream f(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!f) throw std::runtime_error("npy::save: cannot open " + path);

    f.write("\x93NUMPY", 6);
    f.put(2); f.put(0);  // version 2.0

    std::string header = build_header("<i4", false, shape);
    uint32_t hlen = static_cast<uint32_t>(header.size());
    f.write(reinterpret_cast<const char*>(&hlen), 4);
    f.write(header.data(), header.size());

    size_t n = 1;
    for (auto d : shape) n *= d;
    f.write(reinterpret_cast<const char*>(data), n * sizeof(int));
}

inline void save(const std::string& path, const unsigned char* data,
                 const std::vector<size_t>& shape) {
    std::ofstream f(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!f) throw std::runtime_error("npy::save: cannot open " + path);

    f.write("\x93NUMPY", 6);
    f.put(2); f.put(0);

    std::string header = build_header("|u1", false, shape);
    uint32_t hlen = static_cast<uint32_t>(header.size());
    f.write(reinterpret_cast<const char*>(&hlen), 4);
    f.write(header.data(), header.size());

    size_t n = 1;
    for (auto d : shape) n *= d;
    f.write(reinterpret_cast<const char*>(data), n);
}

// Read NPY file, returns (data_ptr, shape). Caller must delete[] data_ptr.
struct Array {
    std::vector<size_t> shape;
    std::string descr;
    bool fortran_order;
    std::vector<char> raw;
};

inline Array load(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::in);
    if (!f) throw std::runtime_error("npy::load: cannot open " + path);

    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy::load: not an NPY file: " + path);

    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t h;
        f.read(reinterpret_cast<char*>(&h), 2);
        hlen = h;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }

    std::string header(hlen, '\0');
    f.read(&header[0], hlen);

    // Parse header dict — minimal: look for 'descr', 'fortran_order', 'shape'
    Array arr;

    auto extract = [&](const std::string& key) -> std::string {
        size_t pos = header.find("'" + key + "':");
        if (pos == std::string::npos) return "";
        pos = header.find(":", pos) + 1;
        while (pos < header.size() && (header[pos] == ' ' || header[pos] == '\t'))
            pos++;
        if (header[pos] == '\'') {
            pos++;
            size_t end = header.find('\'', pos);
            return header.substr(pos, end - pos);
        } else if (header[pos] == 'T') return "True";
        else if (header[pos] == 'F') return "False";
        else if (header[pos] == '(') {
            size_t end = header.find(')', pos);
            return header.substr(pos, end - pos + 1);
        }
        return "";
    };

    arr.descr = extract("descr");
    arr.fortran_order = (extract("fortran_order") == "True");

    std::string shape_str = extract("shape");
    // parse "(d0, d1, ...)" or "(d0,)"
    if (!shape_str.empty() && shape_str.front() == '(') {
        std::istringstream ss(shape_str.substr(1, shape_str.size() - 2));
        std::string token;
        while (std::getline(ss, token, ',')) {
            // trim spaces
            size_t s = 0;
            while (s < token.size() && token[s] == ' ') s++;
            if (s < token.size()) {
                arr.shape.push_back(std::stoull(token.substr(s)));
            }
        }
    }

    size_t nbytes = 1;
    size_t elemsize = 1;
    if (arr.descr == "<i4" || arr.descr == ">i4" || arr.descr == "|i4") elemsize = 4;
    else if (arr.descr == "<f4" || arr.descr == ">f4") elemsize = 4;
    else if (arr.descr == "<f8" || arr.descr == ">f8") elemsize = 8;
    for (auto d : arr.shape) nbytes *= d;
    nbytes *= elemsize;

    arr.raw.resize(nbytes);
    f.read(arr.raw.data(), nbytes);

    return arr;
}

}  // namespace npy
