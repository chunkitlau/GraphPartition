#include "graph.hpp"

enum log_level_set {off = 0, fatal = 1, error = 2, warn = 3, info = 4, debug = 5, trace = 6};

enum log_level_set log_level_graph = debug;

// read graph from file
Graph::Graph(const std::string &input_folder) {
  node_table_ = ReadTable(input_folder + "/node_table");
  edge_table_ = ReadTable(input_folder + "/edge_table");
  train_array_ = ReadArray(input_folder + "/train_table");
  val_array_ = ReadArray(input_folder + "/val_table");
  test_array_ = ReadArray(input_folder + "/test_table");

  // creat edge_vector_map_ from edge_table_ 
  for (auto row: edge_table_.my_matrix()) {
    edge_vector_map_[row[0]].push_back(row[1]);
	}

  // Broadcast ID from each node in the set
  for (auto item: Merge(train_array_, val_array_, test_array_).my_vector()) {
    node_ID_map_[item] = item;
	}
  for (auto item: Merge(train_array_, val_array_, test_array_).my_vector()) {
    if (log_level_graph >= debug) printf("DEBUG: broadcast node %s\n", item.c_str());
    Broadcast(item);
	}
}

// Broadcast ID k-hop from the node vertex
// For each vertex v in these sets, 
// v obtains a unique ID and then broadcasts the ID 
// to its K-hop neighbors being visited by BFS.
void Graph::Broadcast(const std::string &vertex) {
  std::queue<std::pair<std::string,int> > queue;
  queue.push(make_pair(vertex, k_hop));
  while (!queue.empty()) {
    std::pair<std::string,int> front = queue.front();
    queue.pop();
    // Note that each vertex only keeps the first block ID it receives. 
    if (node_ID_map_.find(front.first) == node_ID_map_.end()) {
      node_ID_map_[front.first] = vertex;
      if (log_level_graph >= debug) printf("DEBUG: assign node %s to block %s\n", front.first.c_str(), vertex.c_str());
    }
    if (!front.second) continue;
    for (auto node: edge_vector_map_[front.first]) {
      queue.push(make_pair(node, front.second-1));
    }
  }
}

// construct neighborhood block from graph
// In order to keep the locality of data access, 
// our partitioning algorithm first constructs a neighborhood block 
// for each vertex in the training, validation and test sets.
std::vector<Block> Graph::ConstructNeighborhoodBlock() {
  // All the vertices with the same block ID will then form a neighborhood block. 
  std::map<std::string,Block>  ID_block_map;
  // Assign nodes to corresponding blocks
  for (auto row: node_table_.my_matrix()) {
    // Assume that the ID of node that has not received the broadcast is its own
    if (node_ID_map_.find(row[0]) == node_ID_map_.end()) {
      node_ID_map_[row[0]] = row[0];
    }
    ID_block_map[node_ID_map_[row[0]]].AddNode(row);
	}
  
  // Assign edges to corresponding blocks
  for (auto row: edge_table_.my_matrix()) {
    ID_block_map[node_ID_map_[row[0]]].AddEdge(row);
	}
  
  // Assign train to corresponding blocks
  for (auto item: train_array_.my_vector()) {
    ID_block_map[node_ID_map_[item]].AddTrain(item);
	}
  
  // Assign val to corresponding blocks
  for (auto item: val_array_.my_vector()) {
    ID_block_map[node_ID_map_[item]].AddVal(item);
	}
  
  // Assign test to corresponding blocks
  for (auto item: test_array_.my_vector()) {
    ID_block_map[node_ID_map_[item]].AddTest(item);
	}

  // We sort the blocks in descending order of their sizes and 
  // then start the assignment from the largest block.
  std::vector<Block> block_vector;
  for (auto item: ID_block_map) {
    block_vector.push_back(item.second);
	}
  sort(block_vector.begin(), block_vector.end(), CmpByBlockNodeSize);
  return block_vector;
}

// CrossEdge between partition and block
int Partition::CrossEdge(const Block &block) {
  int count = 0;
  // count edges from block to partition
  for (auto row: block.my_edge_table().my_matrix()) {
    count += IsInNodeSet(row[1]);
	}
  
  // count edges from partition to block;
  for (auto row: block.my_node_table().my_matrix()) {
    count += IsInEdgeDstSet(row[0]);
	}

  return count;
}

// add block to partition
void Partition::AddBlock(Block block) {
  // Add block nodes to corresponding partition
  for (auto row: block.my_node_table().my_matrix()) {
    node_table_.AddRow(row);
    node_set_.insert(row[0]);
	}

  // Add block edges to corresponding partition
  for (auto row: block.my_edge_table().my_matrix()) {
    edge_table_.AddRow(row);
    edge_dst_set_.insert(row[1]);
	}

  // Add block trains to corresponding partition
  for (auto item: block.my_train_array().my_vector()) {
    train_array_.AddItem(item);
	}

  // Add block trains to corresponding partition
  for (auto item: block.my_val_array().my_vector()) {
    val_array_.AddItem(item);
	}

  // Add block trains to corresponding partition
  for (auto item: block.my_test_array().my_vector()) {
    test_array_.AddItem(item);
	}
}

// Set header using graph
void Partition::SetHeader(const Graph &graph) {
  node_table_.set_header(graph.my_node_table().my_header());
  edge_table_.set_header(graph.my_edge_table().my_header());
  train_array_.set_header(graph.my_train_array().my_header());
  val_array_.set_header(graph.my_val_array().my_header());
  test_array_.set_header(graph.my_test_array().my_header());
}

// We sort the blocks in descending order of their sizes
bool CmpByBlockNodeSize(Block &left, Block& right) {
  return left.MyNodeSize() > right.MyNodeSize();
}