# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------

class QaicKvHandOffMetrics():
    """
    Class to hold metrics for kv_handoff
    """
    def __init__(self):
        self._num_kv_handoff = 0
        self._num_kv_handoff_failed = 0
        self._num_kv_handoff_success = 0
        self._num_kv_handoff_success_no_data = 0


import threading
import time
import random
import signal
import sys
from prometheus_client import start_http_server, Histogram, Counter

# Prometheus metrics
REQUEST_PULL_RATE = Counter('request_pull_rate', 'Rate of requests pulled per client', ['client_id'])
REQUEST_PUSH_RATE = Counter('request_push_rate', 'Rate of requests pushed per client', ['client_id'])
REQUEST_PULL_HISTOGRAM = Histogram('request_pull_histogram', 'Histogram of request pull rates per client', ['client_id'])
REQUEST_PUSH_HISTOGRAM = Histogram('request_push_histogram', 'Histogram of request push rates per client', ['client_id'])

class MetricsStatManager:
    def __init__(self, client_ids):
        self.pull_rates = {cid: 0 for cid in client_ids}
        self.push_rates = {cid: 0 for cid in client_ids}
        self.client_ids = client_ids
        self.lock = threading.Lock()
        self.running = True
    
    def record_pull_rate(self, client_id):
        with self.lock:
            self.pull_rates[client_id] += 1
            REQUEST_PULL_RATE.labels(client_id=client_id).inc()
            REQUEST_PULL_HISTOGRAM.labels(client_id=client_id).observe(self.pull_rates[client_id])

    def record_push_rate(self, client_id):
        with self.lock:
            self.push_rates[client_id] += 1
            REQUEST_PUSH_RATE.labels(client_id=client_id).inc()
            REQUEST_PUSH_HISTOGRAM.labels(client_id=client_id).observe(self.push_rates[client_id])
    
    def print_pull_rates(self):
        while self.running:
            time.sleep(2)
            with self.lock:
                print("Pull rates:")
                for client_id in self.client_ids:
                    print(f"Client {client_id}: {self.pull_rates[client_id]} requests")
    
    def stop(self):
        self.running = False

def graceful_exit(manager):
    def handler(signum, frame):
        print("\nShutting down gracefully...")
        manager.stop()
        sys.exit(0)
    return handler

if __name__ == "__main__":
    num_clients = int(input("Enter number of clients: "))
    client_ids = [f"client{i+1}" for i in range(num_clients)]
    
    manager = MetricsStatManager(client_ids)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, graceful_exit(manager))
    signal.signal(signal.SIGTERM, graceful_exit(manager))
    
    # Start Prometheus server
    start_http_server(8001)
    
    # Start background thread
    threading.Thread(target=manager.print_pull_rates, daemon=True).start()
    
    # Simulate server activity
    while True:
        cid = random.choice(client_ids)
        manager.record_pull_rate(cid)
        manager.record_push_rate(cid)
        time.sleep(random.uniform(0.1, 1))
