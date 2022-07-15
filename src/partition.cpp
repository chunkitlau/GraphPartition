#include "partition.hpp"

// Assign block using algorithm 2
std::vector<Partition> AssignBlock(const std::vector<Block> &blocks, int partition_num, double alpha_div_Ctrain, double beta_div_Cval, double gamma_div_Ctest) {
  std::vector<Partition> partitions(partition_num);
  std::vector<double> CE(partition_num), BS(partition_num);
  for (int i = 0; i < blocks.size(); ++i) {
    for (int j = 0; j < partition_num; ++j) {
        if (partitions[j].MyNodeSize()) {
          CE[j] = 1.0 * partitions[j].CrossEdge(blocks[i]) / partitions[j].MyNodeSize();
        }
        else {
          CE[j] = 0;
        }
        BS[j] = (1 - alpha_div_Ctrain * partitions[j].MyTrainSize()
                   - alpha_div_Ctrain * partitions[j].MyValSize()
                   - gamma_div_Ctest * partitions[j].MyTestSize());
        if (log_level >= debug) printf("DEBUG: i = %d CE %d %lf BS %d %lf MyNodeSize %d CrossEdge %d \n", i, j, CE[j], j, BS[j], partitions[j].MyNodeSize(), partitions[j].CrossEdge(blocks[i]));
    }
    int x = 0;
    for (int j = 1; j < partition_num; ++j) {
      if (CE[j] * BS[j] > CE[x] * BS[x] + eps) {
        x = j;
      }
    }
    partitions[x].AddBlock(blocks[i]);
    if (log_level >= debug) printf("DEBUG: assign block %d to partition %d\n", i, x);
	}
  return partitions;
}

// Generate metadata from partitions
std::vector<std::pair<std::string,int> > GenerateMetadata(std::vector<Partition> partitions) {
  std::vector<std::pair<std::string,int> > metadata;
  for (int k = 0; k < partitions.size(); ++k) {
    for (auto row: partitions[k].my_node_table().my_matrix()) {
      metadata.push_back(make_pair(row[0], k));
    }
  }
  return metadata;
}

// main function
int main(int argc,char *argv[]) {
  // Check the validity of command line arguments
  if (argc < 6) {
    printf("Command: ./partition input_folder output_folder partition_num alpha beta gamma");
    return 0;
  }

  // Extract command line arguments
  std::string input_folder(argv[1]),
              output_folder(argv[2]);
  int partition_num = atoi(argv[3]);
  if (!partition_num) {
    if (log_level >= error) printf("ERROR: partition_num = 0\n");
    return 0;
  }
  double alpha = atof(argv[4]),
        beta = atof(argv[5]),
        gamma = atof(argv[6]);

  // read graph from file
  if (log_level >= info) printf("INFO: reading graph from file\n");
  Graph graph(input_folder);

  // construct neighborhood block from graph
  if (log_level >= info) printf("INFO: constructing neighborhood block from graph\n");
  std::vector<Block> blocks = graph.ConstructNeighborhoodBlock();

  // Assign block using algorithm 2
  if (log_level >= info) printf("INFO: assigning block using algorithm 2\n");
  double alpha_div_Ctrain = alpha * graph.MyTrainSize() / partition_num,
        beta_div_Cval = beta * graph.MyValSize() / partition_num,
        gamma_div_Ctest = gamma * graph.MyTestSize() / partition_num;
  std::vector<Partition> partitions = AssignBlock(blocks, partition_num, alpha_div_Ctrain, beta, gamma);
  // Set partition header using graph
  for (auto &partition: partitions) {
    partition.SetHeader(graph);
  }

  // Generate metadata and header for partitions
  if (log_level >= info) printf("INFO: generating metadata and header for partitions\n");
  std::vector<std::pair<std::string,int> > metadata = GenerateMetadata(partitions);
  std::pair<std::string,std::string > metadata_header = make_pair(graph.my_node_table().my_header()[0], "partition-id:int64");
  
  // Write metadata to file
  if (log_level >= info) printf("INFO: writing metadata to file\n");
  WriteMetadata(output_folder, metadata_header, metadata);

  // Write partitions to file
  if (log_level >= info) printf("INFO: writing partitions to file\n");
  WritePartitions(output_folder, partitions);
  return 0;  
}