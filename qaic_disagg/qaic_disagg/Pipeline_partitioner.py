# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""Utility to generate MDP files."""
import json
import sys
import re

def extract_layer_num(node, pattern):
    match = re.search(pattern, node)
    if match:
        return int(match.group(1))
    return None

def get_layer_pattern(node):
    if "/model/language_model/" in node:
        return r"/model/language_model/layers\.(\d+)/"
    elif '/transformer/' in node:
        return r"/transformer/h\.(\d+)/"
    return r"/model/layers\.(\d+)/"

def main(input_json, output_json, num_devices, num_partitions, layers_per_partition):
    with open(input_json, "r") as f:
        data = json.load(f)

    # Flatten all nodes from all partitions in order
    all_nodes = []
    pattern = None
    for part in data["partitions"]:
        if pattern is None and part["nodeList"]:
            pattern = get_layer_pattern(part["nodeList"][0])
        all_nodes.extend(part["nodeList"])

    # Prepare partitions
    partitions = [[] for _ in range(num_partitions)]

    # Find max layer
    max_layer = -1
    for node in all_nodes:
        layer_num = extract_layer_num(node, pattern)
        if layer_num is not None and layer_num > max_layer:
            max_layer = layer_num

    # Assign nodes to partitions, keeping order and grouping special nodes with their surrounding layer
    last_layer_num = 0
    for node in all_nodes:
        layer_num = extract_layer_num(node, pattern)
        if layer_num is not None:
            last_layer_num = layer_num
        partition_idx = min(last_layer_num // layers_per_partition, num_partitions - 1)
        partitions[partition_idx].append(node)

    # Assign devices to partitions
    device_ids = list(range(num_devices))
    devices_per_partition = num_devices // num_partitions
    partition_objs = []
    for i, node_list in enumerate(partitions):
        assigned_devices = device_ids[i * devices_per_partition : (i + 1) * devices_per_partition]
        partition_objs.append({
            "name": f"Partition{i}",
            "nodeList": node_list,
            "devices": [
                {"deviceId": dev_id, "numCores": 16} for dev_id in assigned_devices
            ]
        })

    # Update connections
    data["connections"] = [{
        "devices": device_ids,
        "type": "p2p"
    }]
    data["partitions"] = partition_objs

    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python Pipeline_partitioner.py <input.json> <output.json> <num_devices> <num_partitions> <layers_per_partition>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
