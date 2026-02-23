import flwr as fl


def main():
    # Define strategy: Aggregates weights from connected clients
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=1,  # Min clients to start training (set to 1 for testing)
        min_available_clients=1  # Wait until 1 client is connected
    )

    print("Starting FL Server... Waiting for RPi...")

    # Start the server on Port 8080
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),  # Run 3 rounds of training
        strategy=strategy
    )


if __name__ == "__main__":
    main()