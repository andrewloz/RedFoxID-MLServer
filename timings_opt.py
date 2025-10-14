import time, grpc
from prometheus_client import Histogram, Counter, start_http_server


from config import config
if bool(int(config()["InferenceServer"].get("Profiling", "0"))):
    RPC_LATENCY = Histogram(
        "grpc_server_latency_seconds", "gRPC server latency by method",
        ["grpc_type", "method"],
        buckets=[0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]
    )
    RPC_ERRORS = Counter("grpc_server_errors_total", "gRPC server errors", ["method", "code"])
    PHASE_LATENCY = Histogram(
        "inference_phase_seconds", "Latency of internal phases",
        ["phase", "model"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2]
    )

    def start_metrics_server(port=8000):
        start_http_server(port)

    def phase_observe(phase: str, model: str, dur: float):
        PHASE_LATENCY.labels(phase, model).observe(dur)

    class TimingsInterceptor(grpc.ServerInterceptor):
        def intercept_service(self, continuation, handler_call_details):
            method = handler_call_details.method
            handler = continuation(handler_call_details)
            if handler is None: return None

            def wrap(kind, fn):
                def wrapped(*a, **k):
                    t0 = time.perf_counter()
                    try:
                        return fn(*a, **k)
                    except grpc.RpcError as e:
                        RPC_ERRORS.labels(method, e.code().name).inc()
                        raise
                    finally:
                        RPC_LATENCY.labels(kind, method).observe(time.perf_counter() - t0)
                return wrapped

            if handler.unary_unary:
                return handler._replace(unary_unary=wrap("unary_unary", handler.unary_unary))
            if handler.unary_stream:
                return handler._replace(unary_stream=wrap("unary_stream", handler.unary_stream))
            if handler.stream_unary:
                return handler._replace(stream_unary=wrap("stream_unary", handler.stream_unary))
            if handler.stream_stream:
                return handler._replace(stream_stream=wrap("stream_stream", handler.stream_stream))
            return handler
else:
    # No-ops in OFF mode (literally zero hot-path work)
    TimingsInterceptor = None
    def start_metrics_server(port=8000): ...
    def phase_observe(phase: str, model: str, dur: float): ...