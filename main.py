import subprocess

if __name__ == "__main__":
    # Start server
    server_process = subprocess.Popen(["python", "server.py"])

    # Start clients
    client_processes = []
    for _ in range(5):  # Simulate 5 clients
        client_processes.append(subprocess.Popen(["python", "client.py"]))

    # Wait for processes to finish
    server_process.wait()
    for process in client_processes:
        process.wait()