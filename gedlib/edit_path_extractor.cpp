#include "src/env/ged_env.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset_path> <collection_xml> <ged_method>\n";
        return 1;
    }

    std::string dataset_path   = argv[1];
    std::string collection_xml = argv[2];
    std::string method_str     = argv[3];

    // Convert method_str to uppercase for case-insensitive matching.
    std::transform(method_str.begin(), method_str.end(), method_str.begin(), ::toupper);

    // Disallow methods not intended for approximating GED.
    if (method_str == "RING_ML" || method_str == "BIPARTITE_ML") {
        std::cerr << "Method " << method_str << " should not be used for approximating GED." << std::endl;
        return 1;
    }

    // Map method string to GEDMethod enumeration.
    std::unordered_map<std::string, ged::Options::GEDMethod> method_map = {
        {"BRANCH",              ged::Options::GEDMethod::BRANCH},
        {"BRANCH_FAST",         ged::Options::GEDMethod::BRANCH_FAST},
        {"BRANCH_TIGHT",        ged::Options::GEDMethod::BRANCH_TIGHT},
        {"BRANCH_UNIFORM",      ged::Options::GEDMethod::BRANCH_UNIFORM},
        {"BRANCH_COMPACT",      ged::Options::GEDMethod::BRANCH_COMPACT},
        {"PARTITION",           ged::Options::GEDMethod::PARTITION},
        {"HYBRID",              ged::Options::GEDMethod::HYBRID},
        {"RING",                ged::Options::GEDMethod::RING},
        {"ANCHOR_AWARE_GED",    ged::Options::GEDMethod::ANCHOR_AWARE_GED},
        {"WALKS",               ged::Options::GEDMethod::WALKS},
        {"IPFP",                ged::Options::GEDMethod::IPFP},
        {"BIPARTITE",           ged::Options::GEDMethod::BIPARTITE},
        {"SUBGRAPH",            ged::Options::GEDMethod::SUBGRAPH},
        {"NODE",                ged::Options::GEDMethod::NODE},
        {"REFINE",              ged::Options::GEDMethod::REFINE},
        {"BP_BEAM",             ged::Options::GEDMethod::BP_BEAM},
        {"SIMULATED_ANNEALING", ged::Options::GEDMethod::SIMULATED_ANNEALING},
        {"HED",                 ged::Options::GEDMethod::HED},
        {"STAR",                ged::Options::GEDMethod::STAR}
    };

    if (method_map.find(method_str) == method_map.end()) {
        std::cerr << "Invalid GED method: " << method_str << std::endl;
        return 1;
    }
    ged::Options::GEDMethod method = method_map[method_str];

    // Determine bound type based on the method.
    std::unordered_set<std::string> lowerBoundMethods = {
        "BRANCH", "BRANCH_FAST", "BRANCH_TIGHT", "BRANCH_UNIFORM",
        "BRANCH_COMPACT", "PARTITION", "HYBRID", "ANCHOR_AWARE_GED",
        "SIMULATED_ANNEALING", "HED", "BIPARTITE", "NODE", "STAR"
    };
    std::unordered_set<std::string> upperBoundMethods = {
        "RING", "WALKS", "IPFP", "SUBGRAPH", "REFINE", "BP_BEAM"
    };

    bool use_lower_bound = lowerBoundMethods.find(method_str) != lowerBoundMethods.end();

    // Create and initialize the GED environment.
    ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> ged_env;
    std::vector<ged::GEDGraph::GraphID> all_ids = ged_env.load_gxl_graphs(dataset_path, collection_xml);

    if (all_ids.size() != 2) {
        std::cerr << "Error: dataset must contain exactly 2 graphs." << std::endl;
        return 1;
    }

    ged::GEDGraph::GraphID origId1 = all_ids[0];
    ged::GEDGraph::GraphID origId2 = all_ids[1];

    auto ex1 = ged_env.get_graph(origId1, true, true, true);
    auto ex2 = ged_env.get_graph(origId2, true, true, true);

    // Build exchange graphs.
    ged::GEDGraph::GraphID newId1 = ged_env.load_exchange_graph(
        ex1,
        ged::undefined(),
        ged::Options::ExchangeGraphType::ADJ_LISTS,
        "temp1",
        "temp_class1"
    );
    ged::GEDGraph::GraphID newId2 = ged_env.load_exchange_graph(
        ex2,
        ged::undefined(),
        ged::Options::ExchangeGraphType::ADJ_LISTS,
        "temp2",
        "temp_class2"
    );

    ged_env.set_edit_costs(ged::Options::EditCosts::CONSTANT);
    ged_env.init();

    // Set the chosen GED method with additional options (e.g. thread count).
    ged_env.set_method(method, "--threads 8");
    ged_env.init_method();

    // Run the GED method on the specified graph pair.
    ged_env.run_method(newId1, newId2);

    // Compute the graph edit distance using the appropriate bound.
    double gedCost = use_lower_bound ?
                     ged_env.get_lower_bound(newId1, newId2) :
                     ged_env.get_upper_bound(newId1, newId2);

    // Retrieve the node mapping.
    ged::NodeMap nodeMap = ged_env.get_node_map(newId1, newId2);
    auto forward_map = nodeMap.get_forward_map();

    // Create a JSON object for the node mapping.
    json node_map_json = json::object();
    for (std::size_t i = 0; i < forward_map.size(); ++i) {
        if (forward_map[i] != ged::GEDGraph::undefined_node()) {
            node_map_json[std::to_string(i)] = forward_map[i];
        }
    }

    // Build the final JSON output.
    json output;
    output["ged_method"] = method_str;
    output["graph_edit_distance"] = gedCost;
    output["node_map"] = node_map_json;

    std::cout << output.dump(2) << std::endl;

    return 0;
}