import flwr as fl

def start_server():
    # Define the server configuration
    server_config = fl.server.ServerConfig(num_rounds=3)

    # Start the Flower server
    fl.server.start_server(server_address="127.0.0.1:8080", config=server_config)

if __name__ == "__main__":
    start_server()