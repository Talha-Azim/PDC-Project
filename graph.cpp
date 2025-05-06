#include "include/graph.hpp"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>

void load_csr(const std::string& edge_list_file, Graph& G, bool weighted, int rank) {
    if (rank == 0) {
        // Open the edge list file
        std::ifstream in(edge_list_file);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open the edge list file: " + edge_list_file);
        }

        std::vector<std::pair<int, int>> edges;
        std::string line;
        int max_vertex = 0;

        // Read and parse the edge list
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue; // Ignore empty lines and comments

            std::istringstream iss(line);
            int u, v;
            if (!(iss >> u >> v)) continue; // Parse edge
            u--; v--; // Use 0-based indexing

            // Add edges for an undirected graph
            edges.emplace_back(u, v);
            edges.emplace_back(v, u);

            // Track the highest vertex index
            max_vertex = std::max(max_vertex, std::max(u, v));
        }
        in.close();

        // Set up graph's metadata
        G.num_vertices = max_vertex + 1;
        G.num_edges = edges.size();

        // Sort edges to prepare for CSR conversion
        std::sort(edges.begin(), edges.end());

        // Initialize the CSR arrays
        G.row_ptr.resize(G.num_vertices + 1, 0);  // row_ptr has num_vertices + 1 elements
        G.col_idx.resize(G.num_edges);
        G.weights.resize(G.num_edges);

        // Count edges for each vertex
        for (const auto& edge : edges) {
            G.row_ptr[edge.first + 1]++;
        }

        // Convert counts to prefix sum for row_ptr
        for (int i = 1; i <= G.num_vertices; ++i) {
            G.row_ptr[i] += G.row_ptr[i - 1];
        }

        // Assign column indices and weights for each edge
        std::vector<int> current_pos = G.row_ptr;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(1.0, 10.0);  // For generating random weights

        for (const auto& edge : edges) {
            int pos = current_pos[edge.first]++;
            G.col_idx[pos] = edge.second;  // Destination vertex of the edge
            G.weights[pos] = weighted ? dis(gen) : 1.0;  // Random weight if weighted, else default to 1
        }

        std::cout << "Rank 0: Successfully loaded CSR graph with " 
                  << G.num_vertices << " vertices and " 
                  << G.num_edges << " edges." << std::endl;
    }

    // Share graph metadata with other processes
    MPI_Bcast(&G.num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&G.num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize arrays for non-root processes
    if (rank != 0) {
        G.row_ptr.resize(G.num_vertices + 1);
        G.col_idx.resize(G.num_edges);
        G.weights.resize(G.num_edges);
    }

    // Broadcast the CSR arrays to all processes
    MPI_Bcast(G.row_ptr.data(), G.num_vertices + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(G.col_idx.data(), G.num_edges, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(G.weights.data(), G.num_edges, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
