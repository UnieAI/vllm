# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
import threading
import argparse
import signal
import re
import time
import zmq
import logging
from logging import DEBUG
from rich import print
from qaic_disagg.storage.key_value_store import QaicKVStore, KVCacheEntry
from qaic_disagg.kv_handoff.protocol import  QaicKvHandOffPutReq, QaicKvHandOffGetReq, ReqType, QaicKvHandOffGetResp, QaicBufferType, QaicKvHandOffReqType
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.utils import is_valid_ipv6_address
from qaic_disagg.utils import zmq_socket_ctx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class QaicCacheServer:
    def __init__(self, host, port, size):
        """
        Initialize the QaicCacheServer with the specified host and port.
        """
        logger.info("Initializing QaicCacheServer on {}:{}".format(host, port))
        self.host = host
        self.is_ipv6 = False
        if is_valid_ipv6_address(host):
            self.host = '[' + self.host + ']'
            self.is_ipv6 = True

        self.port = port
        self.max_size = size
        self.data_store = QaicKVStore()
        self.decoder_put = MsgpackDecoder(QaicKvHandOffPutReq)
        self.decoder_get = MsgpackDecoder(QaicKvHandOffGetReq)
        self.encoder = MsgpackEncoder(QaicKvHandOffGetResp)
        self.close_event = threading.Event()

    def process_client_request(self, ipc_path):
        """
        Process client requests using the specified IPC path.
        """
        with zmq_socket_ctx(ipc_path, zmq.ROUTER, is_ipv6=self.is_ipv6) as socket:
            while True:
                try:
                    (identity, cmd, msg) = socket.recv_multipart(copy=False)

                    if not identity or not cmd:
                        break
                    cmd = QaicKvHandOffReqType(bytes(cmd.buffer))
                    resp_payload = b'\x00'

                    if cmd == QaicKvHandOffReqType.PUT:
                        try:
                            header = self.decoder_put.decode(msg)
                            if len(self.data_store) <= self.max_size:
                                self.data_store.add(header.key_hash, KVCacheEntry.create_obj(header))
                                resp = QaicKvHandOffReqType.RESP_OK
                            else:
                                resp = QaicKvHandOffReqType.RESP_BUFFER_FULL
                        except Exception as e:
                            logger.warning(f"Error processing PUT request: {e} for identity {identity}")
                            resp = QaicKvHandOffReqType.RESP_ERROR

                    elif cmd == QaicKvHandOffReqType.GET:
                        try:
                            header = self.decoder_get.decode(msg)
                            if header.key_hash in self.data_store:
                                resp = QaicKvHandOffReqType.RESP_OK
                                resp_payload = self.data_store.get(header.key_hash).get_pkt_to_send()
                                resp_payload = self.encoder.encode(resp_payload)[0]
                            else:
                                resp = QaicKvHandOffReqType.RESP_NOT_FOUND
                        except Exception as e:
                            logger.warning(f"Error processing GET request: {e} for identity {identity}")
                            resp = QaicKvHandOffReqType.RESP_ERROR
                    else:
                        resp = QaicKvHandOffReqType.RESP_INVALID_CMD
                    socket.send_multipart((identity, resp.value, resp_payload))
                except Exception as e:
                    logger.error(f"Error processing request: {e} for identity {identity} and for cmd {cmd}")

        logger.info("Client request processing stopped.")

    def signal_handler(self, *_):
        """
        Signal handler to stop the server.
        """
        logger.info("Received signal to stop the server.")
        self.close_event.set()
        if hasattr(self,'data_store'):
            del self.data_store

    def run_server(self):
        """
        Run the server.
        """

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        logger.info("Server started.")
        #print(f"Server started at {self.host}:{self.port}")
        ipc_path = f"tcp://{self.host}:{self.port}"
        _thread = threading.Thread(target=self.process_client_request, args=(ipc_path, ))
        _thread.daemon = True
        _thread.start()
        while not self.close_event.is_set():
            time.sleep(2)

        logger.info("Server stopped.")


def main():
    """
    Main function to start the server.
    """
    parser = argparse.ArgumentParser("QAIC KV hand-off service.")
    parser.add_argument("--host",
                        "-H",
                        type=str,
                        default="localhost",
                        help="Host address")
    parser.add_argument("--port",
                        "-P",
                        type=int,
                        default=8080,
                        help="Port number")
    parser.add_argument("--size",
                        "-S",
                        type=int,
                        default=64,
                        help="Size of the KV Cache store")
    args = parser.parse_args()

    try:
        args.port = int(args.port)
    except ValueError:
        raise argparse.ArgumentTypeError("Port must be an integer.")
    if not 1 <= args.port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")    
    print(args)
    server = QaicCacheServer(args.host, args.port, args.size)
    server.run_server()

if __name__ == "__main__":
    main()