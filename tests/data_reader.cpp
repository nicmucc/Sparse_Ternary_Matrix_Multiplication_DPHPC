
#include "data_reader.h"

#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

SampleData read_sample(const std::string& directory) {

    SampleData data;

    // Read metadata
    std::ifstream metadata_file(directory + "/metadata.json");

    if (!metadata_file)
        throw std::runtime_error("Cannot open metadata.json");

    json metadata;
    metadata_file >> metadata;

    // Read dimensions
    data.M = metadata["config"]["M"].get<std::size_t>();
    data.K = metadata["config"]["K"].get<std::size_t>();
    data.N = metadata["config"]["N"].get<std::size_t>();

    // Read tolerances
    data.atol = metadata["reference"]["atol"].get<double>();
    data.rtol = metadata["reference"]["rtol"].get<double>();

    // Allocate vectors
    data.X.resize(data.M * data.K);
    data.W.resize(data.K * data.N);
    data.bias.resize(data.N);
    data.alpha.resize(1);
    data.expected_linear.resize(data.M * data.N);
    data.expected_prelu.resize(data.M * data.N);

    // Read X
    std::ifstream x_file(directory + "/X.bin", std::ios::binary);

    if (!x_file)
        throw std::runtime_error("Cannot open X.bin");

    x_file.read(
        reinterpret_cast<char*>(data.X.data()),
        data.X.size() * sizeof(float)
    );

    if (!x_file)
        throw std::runtime_error("Error reading X.bin");


    // Read W (stored as int8 in Python)
    std::ifstream w_file(directory + "/W.bin", std::ios::binary);

    if (!w_file)
        throw std::runtime_error("Cannot open W.bin");

    std::vector<int8_t> temp_W(data.K * data.N);

    w_file.read(
        reinterpret_cast<char*>(temp_W.data()),
        temp_W.size() * sizeof(int8_t)
    );

    if (!w_file)
        throw std::runtime_error("Error reading W.bin");

    // Convert int8_t to int
    data.W.assign(temp_W.begin(), temp_W.end());


    // Read bias
    std::ifstream bias_file(directory + "/bias.bin", std::ios::binary);

    if (!bias_file)
        throw std::runtime_error("Cannot open bias.bin");

    bias_file.read(
        reinterpret_cast<char*>(data.bias.data()),
        data.bias.size() * sizeof(float)
    );

    if (!bias_file)
        throw std::runtime_error("Error reading bias.bin");


    // Read alpha
    std::ifstream alpha_file(directory + "/alpha.bin", std::ios::binary);

    if (!alpha_file)
        throw std::runtime_error("Cannot open alpha.bin");

    alpha_file.read(
        reinterpret_cast<char*>(data.alpha.data()),
        data.alpha.size() * sizeof(float)
    );

    if (!alpha_file)
        throw std::runtime_error("Error reading alpha.bin");


    // Read expected linear result
    std::ifstream linear_file(
        directory + "/expected_linear.bin", std::ios::binary
    );

    if (!linear_file)
        throw std::runtime_error("Cannot open expected_linear.bin");

    linear_file.read(
        reinterpret_cast<char*>(data.expected_linear.data()),
        data.expected_linear.size() * sizeof(float)
    );

    if (!linear_file)
        throw std::runtime_error("Error reading expected_linear.bin");


    // Read expected PReLU result
    std::ifstream prelu_file(
        directory + "/expected_prelu.bin", std::ios::binary
    );

    if (!prelu_file)
        throw std::runtime_error("Cannot open expected_prelu.bin");

    prelu_file.read(
        reinterpret_cast<char*>(data.expected_prelu.data()),
        data.expected_prelu.size() * sizeof(float)
    );

    if (!prelu_file)
        throw std::runtime_error("Error reading expected_prelu.bin");

    return data;
}
