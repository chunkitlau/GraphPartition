#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <string>
#include <cstring>
#include <regex>
#include <sys/stat.h> 
#include <sys/types.h>

class Table;
class Array;
class Partition;

// read table from file
Table ReadTable(const std::string &input_filename);

// read array from file
Array ReadArray(const std::string &input_filename);

// write table to file
void WriteTable(std::string output_filename, Table table);

// write array to file
void WriteArray(std::string output_filename, Array array);

// write vector to file
void WriteVector(std::ofstream &file, const std::vector<std::string> &vector);

// merge 3 arrays into 1 array
Array Merge(Array &train_array_, Array &val_array_, Array &test_array_);

// Split the string into a string vector according to pattern
std::vector<std::string> Split(std::string &str, const std::string &pattern);

// Write metadata to file
void WriteMetadata(const std::string &output_folder, const std::pair<std::string,std::string > &metadata_header, const std::vector<std::pair<std::string,int> > &metadata);

// Write partitions to file
void WritePartitions(const std::string &output_folder, const std::vector<Partition> &partitions);

#endif