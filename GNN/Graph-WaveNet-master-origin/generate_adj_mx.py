import numpy as np
import pickle
import os

def generate_adj_mx_for_metr_la():
    """
    Generate adjacency matrix file for METR-LA dataset.
    METR-LA has 207 sensors/nodes.
    """
    num_nodes = 207
    
    # Create sensor IDs (simple sequential IDs)
    sensor_ids = [str(i) for i in range(num_nodes)]
    
    # Create sensor ID to index mapping
    sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    
    # Create a simple adjacency matrix
    # For demonstration, we'll create an identity matrix (each node connected to itself)
    # In a real scenario, this would be based on actual geographical distances
    adj_mx = np.eye(num_nodes, dtype=np.float32)
    
    # You can also create a more connected graph if needed
    # For example, connect each node to its k nearest neighbors
    # For now, let's add some random connections to make it more realistic
    np.random.seed(42)  # For reproducibility
    
    # Add some random connections (sparse connectivity)
    for i in range(num_nodes):
        # Connect to a few random neighbors
        num_connections = np.random.randint(1, 6)  # 1-5 connections per node
        neighbors = np.random.choice(num_nodes, size=num_connections, replace=False)
        for neighbor in neighbors:
            if i != neighbor:
                # Add bidirectional connection with random weight
                weight = np.random.uniform(0.1, 1.0)
                adj_mx[i, neighbor] = weight
                adj_mx[neighbor, i] = weight
    
    # Ensure diagonal is 1 (self-connection)
    np.fill_diagonal(adj_mx, 1.0)
    
    return sensor_ids, sensor_id_to_ind, adj_mx

def save_adj_mx(sensor_ids, sensor_id_to_ind, adj_mx, output_path):
    """Save the adjacency matrix data to pickle file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as pickle file
    with open(output_path, 'wb') as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj_mx), f, protocol=2)
    
    print(f"Adjacency matrix saved to {output_path}")
    print(f"Matrix shape: {adj_mx.shape}")
    print(f"Number of sensors: {len(sensor_ids)}")

if __name__ == "__main__":
    # Generate adjacency matrix for METR-LA
    sensor_ids, sensor_id_to_ind, adj_mx = generate_adj_mx_for_metr_la()
    
    # Save to the expected location
    output_path = "data/sensor_graph/adj_mx.pkl"
    save_adj_mx(sensor_ids, sensor_id_to_ind, adj_mx, output_path)
    
    print("Adjacency matrix generation completed!") 