#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include <thread>
#include <mutex>
#include "utils.hpp"

// size of map, support thread number
#define MAP_SIZE_THREAD 8

// K-hop number
extern const int k_hop;

// node ID map mutex
static std::mutex node_ID_map_mutex_[MAP_SIZE_THREAD];

class Block;

// table class
class Table {
private:
  std::vector<std::string> header_;
  std::vector<std::vector<std::string> > matrix_;
public:  
  Table() {}
  Table(const std::vector<std::string> &header, const std::vector<std::vector<std::string> > &matrix) {
    header_ = header;
    matrix_ = matrix;
  }
  std::vector<std::string> my_header() const {
    return header_;
  }
  std::vector<std::vector<std::string> > my_matrix() const {
    return matrix_;
  }
  int MyNodeSize() const {
    return matrix_.size();
  }
  void set_header(const std::vector<std::string> &header) {
    header_ = header;
  }
  void AddRow(const std::vector<std::string> &row) {
    matrix_.push_back(row);
  }
};

// array class
class Array {
private:
  std::vector<std::string> header_;
  std::vector<std::string> vector_;
public:  
  Array() {}
  Array(const std::vector<std::string> &header, const std::vector<std::string> &vector) {
    header_ = header;
    vector_ = vector;
  }
  std::vector<std::string> my_header() const {
    return header_;
  }
  std::vector<std::string> my_vector() const {
    return vector_;
  }
  int MySize() const {
    return vector_.size();
  }
  void set_header(const std::vector<std::string> &header) {
    header_ = header;
  }
  void AddItem(const std::string &item) {
    vector_.push_back(item);
  }
};

// graph class
class Graph {
private:
  // thread set
  enum thread_set {single_thread = 0, multi_thread = 1};
  // thread_level
  enum thread_set thread_level = multi_thread;
  std::map<std::string,std::vector<std::string> > edge_vector_map_;
  std::map<std::string,std::string> node_ID_map_[MAP_SIZE_THREAD];
protected:
  Table node_table_, edge_table_;
  Array train_array_, val_array_, test_array_;
public:  
  Graph() {}
  // read graph from file
  Graph(const std::string &input_folder);
  int MyTrainSize() const {
    return train_array_.MySize();
  }
  int MyValSize() const {
    return val_array_.MySize();
  }
  int MyTestSize() const {
    return test_array_.MySize();
  }
  Table my_node_table() const {
    return node_table_;
  }
  Table my_edge_table() const {
    return edge_table_;
  }
  Array my_train_array() const {
    return train_array_;
  }
  Array my_val_array() const {
    return val_array_;
  }
  Array my_test_array() const {
    return test_array_;
  }
  // Broadcast ID k-hop from the node vertex
  void Broadcast(const std::string &vertex);
  // Broadcast ID k-hop from the node vertex multi thread
  void BroadcastMultiThread(const std::string &vertex);
  // construct neighborhood block from graph
  std::vector<Block> ConstructNeighborhoodBlock();
  // hashing string based on last character
  unsigned int Hashing(const std::string &string) {
    return string[string.length()-1] % MAP_SIZE_THREAD;
  }
};

// block class
class Block: public Graph {
public:  
  int MyNodeSize() const {
    return node_table_.MyNodeSize();
  }
  Table my_node_table() const {
    return node_table_;
  }
  Table my_edge_table() const {
    return edge_table_;
  }
  Array my_train_array() const {
    return train_array_;
  }
  Array my_val_array() const {
    return val_array_;
  }
  Array my_test_array() const {
    return test_array_;
  }
  void AddNode(const std::vector<std::string> &node) {
    node_table_.AddRow(node);
  }
  void AddEdge(const std::vector<std::string> &edge) {
    edge_table_.AddRow(edge);
  }
  void AddTrain(const std::string &train) {
    train_array_.AddItem(train);
  }
  void AddVal(const std::string &val) {
    val_array_.AddItem(val);
  }
  void AddTest(const std::string &test) {
    test_array_.AddItem(test);
  }
};

// partition class
class Partition: public Graph {
private:
  std::set<std::string> node_set_, edge_dst_set_;
public:  
  int MyNodeSize() const {
    return node_table_.MyNodeSize();
  }
  Table my_node_table() const {
    return node_table_;
  }
  // return 1 if node in partition node set
  int IsInNodeSet(const std::string &node) const {
    return node_set_.find(node) != node_set_.end();
  }
  // return 1 if node in partition edge dst set
  int IsInEdgeDstSet(const std::string &node) const {
    return edge_dst_set_.find(node) != edge_dst_set_.end();
  }
  void AddNode(const std::vector<std::string> &node) {
    node_table_.AddRow(node);
  }
  void AddEdge(const std::vector<std::string> &edge) {
    edge_table_.AddRow(edge);
  }
  void AddTrain(const std::string &train) {
    train_array_.AddItem(train);
  }
  void AddVal(const std::string &val) {
    val_array_.AddItem(val);
  }
  void AddTest(const std::string &test) {
    test_array_.AddItem(test);
  }
  // Set header using graph
  void SetHeader(const Graph &graph);
  // CrossEdge between partition and block
  int CrossEdge(const Block &block);
  // add block to partition
  void AddBlock(Block block);
};

// We sort the blocks in descending order of their sizes
bool CmpByBlockNodeSize(Block &left, Block& right);

#endif