#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include "utils.hpp"

// K-hop number
extern const int k_hop;

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
  void AddRow(const std::vector<std::string> &row) {
    matrix_.push_back(row);
  }
  int MyNodeSize() const {
    return matrix_.size();
  }
  void set_header(const std::vector<std::string> &header) {
    header_ = header;
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
  void AddItem(const std::string &item) {
    vector_.push_back(item);
  }
  int MySize() const {
    return vector_.size();
  }
  void set_header(const std::vector<std::string> &header) {
    header_ = header;
  }
};

// graph class
class Graph {
private:
  std::map<std::string,std::vector<std::string> > edge_vector_map_;
  std::map<std::string,std::string> node_ID_map_;
protected:
  Table node_table_, edge_table_;
  Array train_array_, val_array_, test_array_;
public:  
  Graph() {}
  // read graph from file
  Graph(const std::string &input_folder);
  // Broadcast ID k-hop from the node vertex
  void Broadcast(const std::string &vertex);
  // construct neighborhood block from graph
  std::vector<Block> ConstructNeighborhoodBlock();
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
};

// block class
class Block: public Graph {
public:  
  Block() {}
  ~Block() {}
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
};

// partition class
class Partition: public Graph {
private:
  std::set<std::string> node_set_, edge_dst_set_;
public:  
  Partition() {}
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
  int MyNodeSize() const {
    return node_table_.MyNodeSize();
  }
  Table my_node_table() const {
    return node_table_;
  }
  // CrossEdge between partition and block
  int CrossEdge(const Block &block);
  // add block to partition
  void AddBlock(Block block);
  // return 1 if node in partition node set
  int IsInNodeSet(const std::string &node) const {
    return node_set_.find(node) != node_set_.end();
  }
  // return 1 if node in partition edge dst set
  int IsInEdgeDstSet(const std::string &node) const {
    return edge_dst_set_.find(node) != edge_dst_set_.end();
  }
  // Set header using graph
  void SetHeader(const Graph &graph);
};

// We sort the blocks in descending order of their sizes
bool CmpByBlockNodeSize(Block &left, Block& right);

#endif