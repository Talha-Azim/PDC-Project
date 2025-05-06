#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <mpi.h>

struct GraphChunk {
    int total_nodes;
    int local_start;
    int local_end;
    int nodes_owned;
    int edges_owned;
    std::vector<int> row_starts;
    std::vector<int> neighbors;
    std::vector<double> edge_weights;
};

void read_distributed_graph(const char* file, GraphChunk& my_graph, int my_rank, int total_ranks) {
    int max_node_id = 0;
    int edge_count = 0;

    if (my_rank == 0) {
        std::ifstream infile(file);
        if (!infile) {
            std::cerr << "File open failed for metadata!\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::string line;
        while (std::getline(infile, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream parser(line);
            int a, b;
            if (parser >> a >> b) {
                max_node_id = std::max(max_node_id, std::max(a, b));
                edge_count++;
            }
        }
        infile.close();
    }

    MPI_Bcast(&max_node_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    my_graph.total_nodes = max_node_id;

    int nodes_per_proc = (my_graph.total_nodes + total_ranks - 1) / total_ranks;
    my_graph.local_start = my_rank * nodes_per_proc;
    my_graph.local_end = std::min((my_rank + 1) * nodes_per_proc, my_graph.total_nodes);
    my_graph.nodes_owned = my_graph.local_end - my_graph.local_start;

    std::vector<int> edge_count_per_node(my_graph.nodes_owned, 0);
    int local_edge_contrib = 0;

    std::ifstream data(file);
    if (!data) {
        std::cerr << "Unable to open file on process " << my_rank << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string record;
    while (std::getline(data, record)) {
        if (record.empty() || record[0] == '#') continue;
        std::istringstream parser(record);
        int u, v;
        if (!(parser >> u >> v)) continue;
        u--; v--;

        if (u >= my_graph.local_start && u < my_graph.local_end) {
            edge_count_per_node[u - my_graph.local_start]++;
            local_edge_contrib++;
        }
        if (v >= my_graph.local_start && v < my_graph.local_end) {
            edge_count_per_node[v - my_graph.local_start]++;
        }
    }
    data.close();

    my_graph.row_starts.resize(my_graph.nodes_owned + 1, 0);
    my_graph.edges_owned = 0;
    for (int i = 0; i < my_graph.nodes_owned; ++i) {
        my_graph.row_starts[i + 1] = my_graph.row_starts[i] + edge_count_per_node[i];
        my_graph.edges_owned += edge_count_per_node[i];
    }

    std::fill(edge_count_per_node.begin(), edge_count_per_node.end(), 0);

    my_graph.neighbors.resize(my_graph.edges_owned);
    my_graph.edge_weights.resize(my_graph.edges_owned);

    data.open(file);
    if (!data) {
        std::cerr << "Failed to reopen file on process " << my_rank << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::random_device seed;
    std::mt19937 rng(seed());
    std::uniform_real_distribution<> weight_gen(1.0, 10.0);

    while (std::getline(data, record)) {
        if (record.empty() || record[0] == '#') continue;
        std::istringstream parser(record);
        int u, v;
        if (!(parser >> u >> v)) continue;
        u--; v--;

        if (u >= my_graph.local_start && u < my_graph.local_end) {
            int local_u = u - my_graph.local_start;
            int edge_idx = my_graph.row_starts[local_u] + edge_count_per_node[local_u];
            my_graph.neighbors[edge_idx] = v;
            my_graph.edge_weights[edge_idx] = weight_gen(rng);
            edge_count_per_node[local_u]++;
        }

        if (v >= my_graph.local_start && v < my_graph.local_end) {
            int local_v = v - my_graph.local_start;
            int edge_idx = my_graph.row_starts[local_v] + edge_count_per_node[local_v];
            my_graph.neighbors[edge_idx] = u;
            my_graph.edge_weights[edge_idx] = weight_gen(rng);
            edge_count_per_node[local_v]++;
        }
    }
    data.close();

    int total_edges_loaded;
    MPI_Reduce(&local_edge_contrib, &total_edges_loaded, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        std::cout << "Graph loaded: " << my_graph.total_nodes << " nodes, " << total_edges_loaded << " edges.\n";
        if (total_edges_loaded != edge_count) {
            std::cerr << "Warning: Mismatch in edge count. Found " << total_edges_loaded << ", expected " << edge_count << "\n";
        }
    }
}

void run_parallel_delta(GraphChunk& my_graph, int start_node, std::vector<double>& distance, 
                        std::vector<int>& parent_node, int my_rank, int total_ranks) {
    const double DELTA = 10.0;
    std::vector<bool> processed(my_graph.total_nodes, false);
    std::vector<std::vector<int>> buckets(my_graph.total_nodes / DELTA + 1);
    std::vector<double> global_distance(my_graph.total_nodes, std::numeric_limits<double>::infinity());
    std::vector<int> global_parents(my_graph.total_nodes, -1);

    if (start_node >= my_graph.local_start && start_node < my_graph.local_end) {
        distance[start_node] = 0.0;
        parent_node[start_node] = start_node;
        buckets[0].push_back(start_node);
    }

    int curr_bucket = 0;
    while (curr_bucket < buckets.size()) {
        std::vector<int> active_nodes;
        active_nodes.swap(buckets[curr_bucket]);

        while (!active_nodes.empty()) {
            std::vector<int> updated_nodes;
            for (int node : active_nodes) {
                if (processed[node]) continue;
                processed[node] = true;

                if (node >= my_graph.local_start && node < my_graph.local_end) {
                    int local_node = node - my_graph.local_start;
                    for (int i = my_graph.row_starts[local_node]; i < my_graph.row_starts[local_node + 1]; ++i) {
                        int neighbor = my_graph.neighbors[i];
                        double weight = my_graph.edge_weights[i];
                        double new_dist = distance[node] + weight;

                        if (new_dist < global_distance[neighbor]) {
                            global_distance[neighbor] = new_dist;
                            global_parents[neighbor] = node;
                            int new_bucket = static_cast<int>(new_dist / DELTA);
                            if (new_bucket < buckets.size() && neighbor >= my_graph.local_start && neighbor < my_graph.local_end) {
                                buckets[new_bucket].push_back(neighbor);
                            }
                        }
                    }
                }
            }

            // Gather updates from all processes
            MPI_Allreduce(MPI_IN_PLACE, global_distance.data(), my_graph.total_nodes, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, global_parents.data(), my_graph.total_nodes, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            for (int i = 0; i < my_graph.total_nodes; ++i) {
                if (global_distance[i] < distance[i]) {
                    distance[i] = global_distance[i];
                    parent_node[i] = global_parents[i];
                    if (!processed[i] && i >= my_graph.local_start && i < my_graph.local_end) {
                        int bkt = static_cast<int>(distance[i] / DELTA);
                        if (bkt < buckets.size()) {
                            buckets[bkt].push_back(i);
                        }
                    }
                }
            }

            active_nodes = buckets[curr_bucket];
        }
        curr_bucket++;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int my_rank, total_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);

    if (argc < 2) {
        if (my_rank == 0) std::cerr << "Usage: " << argv[0] << " <input_graph_file>\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    GraphChunk my_part;
    read_distributed_graph(argv[1], my_part, my_rank, total_procs);

    std::vector<double> shortest_dist(my_part.total_nodes, std::numeric_limits<double>::infinity());
    std::vector<int> path_parent(my_part.total_nodes, -1);

    int source_node = 0;

    double t_start = MPI_Wtime();
    run_parallel_delta(my_part, source_node, shortest_dist, path_parent, my_rank, total_procs);
    double t_end = MPI_Wtime();

    if (my_rank == 0) {
        std::cout << "SSSP (Delta-Stepping) completed in " << (t_end - t_start) << " seconds.\n";
    }

    MPI_Finalize();
    return 0;
}
