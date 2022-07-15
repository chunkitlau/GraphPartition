#ifndef PARTITION_HPP
#define PARTITION_HPP

#include "graph.hpp"
#include "utils.hpp"

// K-hop number
const int k_hop = 1;

const double eps = 1e-6;

enum log_level_set {off = 0, fatal = 1, error = 2, warn = 3, info = 4, debug = 5, trace = 6};

enum log_level_set log_level = debug;

// Assign block using algorithm 2
std::vector<Partition> AssignBlock(const std::vector<Block> &blocks, int partition_num, double alpha_div_Ctrain, double beta_div_Cval, double gamma_div_Ctest);

// Generate metadata from partitions
std::vector<std::pair<std::string,int> > GenerateMetadata(std::vector<Partition> partitions);

#endif