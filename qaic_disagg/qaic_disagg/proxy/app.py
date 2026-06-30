# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------

# NOTE: Make sure all vllm imports are lazy loaded to prevent triggering worker startup timeout in uvicorn
# Ref: https://github.com/Kludex/uvicorn/issues/2506

from fastapi import FastAPI
import uvicorn
from qaic_disagg.proxy.server import parse_args, ProxyServer
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize all components for this worker process
    args = getattr(app.state, "args", None)
    if args is None:
        args = parse_args()
        if args.workers and not args.workers > 1:
            logger.debug(
                "Parsing args again during app lifecycle. This should not happen in single-worker case."
            )
        setup_app_with_args(app, args)
    yield


def create_app():
    """Create the FastAPI app with lifespan handler."""
    return FastAPI(lifespan=lifespan)


def setup_app_with_args(app: FastAPI, args):
    """Store args in app.state and include router."""
    proxy_server = ProxyServer(args)
    app.include_router(proxy_server.proxy_instance.router)
    app.state.args = args


# Only create the app object; actual setup happens in lifespan
app = create_app()


def main():
    args = parse_args()
    setup_app_with_args(app, args)

    from vllm.utils import set_ulimit

    set_ulimit()
    # Start Uvicorn with workers. String format of app is necessary for multi-worker support.
    uvicorn.run(
        "qaic_disagg.proxy.app:app",
        host=args.host,
        port=args.port,
        loop="uvloop",
        workers=args.workers,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        log_level=args.uvicorn_log_level,
        # NOTE: When the 'disable_uvicorn_access_log' value is True,
        # no access log will be output.
        access_log=not args.disable_uvicorn_access_log,
    )


if __name__ == "__main__":
    main()
