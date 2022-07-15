#include "utils.hpp"
#include "graph.hpp"

enum log_level_set {off = 0, fatal = 1, error = 2, warn = 3, info = 4, debug = 5, trace = 6};

enum log_level_set log_level_utils = debug;

// read table from file
Table ReadTable(const std::string &input_filename) {
  std::ifstream file; 
  file.open(input_filename.c_str());
  std::string buf;
  getline(file, buf);
  std::vector<std::string> header = Split(buf, "\\t+");
  std::vector<std::vector<std::string> > matrix;
  while (getline(file, buf)) {
    std::vector<std::string> line = Split(buf, "\\t+");
    matrix.push_back(line);
  }
  Table table(header, matrix);
  return table;
}

// read array from file
Array ReadArray(const std::string &input_filename) {
  std::ifstream file; 
  file.open(input_filename.c_str());
  std::string buf;
  getline(file, buf);
  std::vector<std::string> header = Split(buf, "\\t+");
  std::vector<std::string> vector;
  while (getline(file, buf)) {
    std::vector<std::string> line = Split(buf, "\\t+");
    vector.push_back(line[0]);
  }
  Array array(header, vector);
  return array;
}

// merge 3 arrays into 1 array
Array Merge(Array &train_array_, Array &val_array_, Array &test_array_) {
  // We assume that all arrays have the same header
  std::vector<std::string> vector = train_array_.my_vector(), 
                           val_array = val_array_.my_vector(), 
                           test_array = test_array_.my_vector();
  vector.insert(vector.end(), val_array.begin(), val_array.end());
  vector.insert(vector.end(), test_array.begin(), test_array.end());
  std::vector<std::string> header = train_array_.my_header();
  Array array(header, vector);
  return array;
}

// Split the string into a string vector according to pattern
std::vector<std::string> Split(std::string &str, const std::string &pattern) {
  if (str[str.length()-1] == '\n') str.erase(str.end() - 1);
  if (str[str.length()-1] == '\r') str.erase(str.end() - 1);
  std::regex ws_re(pattern);
  std::vector<std::string> vector(std::sregex_token_iterator(str.begin(), str.end(), ws_re, -1), std::sregex_token_iterator());
  return vector;
}

// Write metadata to file
void WriteMetadata(const std::string &output_folder, const std::pair<std::string,std::string > &metadata_header, const std::vector<std::pair<std::string,int> > &metadata) {
  std::ofstream file; 
  file.open((output_folder + "/metadata").c_str());
  file << metadata_header.first << "\t" << metadata_header.second << std::endl;
  for (auto row: metadata) {
    file << row.first << "\t" << row.second << std::endl;
  }
}

void WritePartitions(const std::string &output_folder, const std::vector<Partition> &partitions) {
  for (int k = 0; k < partitions.size(); ++k) {
    std::string part_folder = output_folder + "/part" + std::to_string(k);
    int isCreate = mkdir(part_folder.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    WriteTable(part_folder + "/node_table", partitions[k].my_node_table());
    WriteTable(part_folder + "/edge_table", partitions[k].my_edge_table());
    WriteArray(part_folder + "/train_table", partitions[k].my_train_array());
    WriteArray(part_folder + "/val_table", partitions[k].my_val_array());
    WriteArray(part_folder + "/test_table", partitions[k].my_test_array());
  }
  
}

// write vector to file
void WriteVector(std::ofstream &file, const std::vector<std::string> &vector) {
  file << vector[0];
  for (int k = 1; k < vector.size(); ++k) {
    file << "\t" << vector[k];
  }
  file << std::endl;
}

// write table to file
void WriteTable(std::string output_filename, Table table) {
  std::ofstream file; 
  file.open(output_filename.c_str());
  WriteVector(file, table.my_header());
  for (auto row: table.my_matrix()) {
    WriteVector(file, row);
  }
}

// write array to file
void WriteArray(std::string output_filename, Array array) {
  std::ofstream file; 
  file.open(output_filename.c_str());
  WriteVector(file, array.my_header());
  for (auto row: array.my_vector()) {
    file << row << std::endl;
  }
}