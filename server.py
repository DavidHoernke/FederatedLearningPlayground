import flwr as fl

def start_server():
    # Initialize the Flower server
    fl.server.start_server(server_address="127.0.0.1:8080", config={"num_rounds": 3})

if __name__ == "__main__":
    start_server()