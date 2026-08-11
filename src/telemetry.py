"""Superlog telemetry bootstrap: OTel traces, logs, and metrics.

Import this BEFORE torch/transformers in any entry point. The endpoint
and project-scoped public ingest token are inlined by design — the
token is write-only (like a Sentry DSN); it cannot read data or touch
the account. See .agents/skills/superlog-onboard for the contract.
"""

import atexit
import logging
import os

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

ENDPOINT = "https://intake.superlog.sh"
PUBLIC_TOKEN = "sl_public_gYAhWySVrnVtRMy5KEKqRWyt4GcDZdnQkaN15Z-jJRw"


def superlog_headers(token: str) -> dict:
    return {"x-api-key": token}


_HEADERS = superlog_headers(PUBLIC_TOKEN)

_attrs = {
    "service.name": "nepali-peft-train",
    "service.version": "0.1.0",
    "deployment.environment.name":
        "colab" if os.environ.get("COLAB_RELEASE_TAG") else "local",
    "vcs.repository.url.full": "https://github.com/subresh-code/nepali-peft",
}
_sha = (os.environ.get("GITHUB_SHA") or os.environ.get("SOURCE_COMMIT")
        or os.environ.get("GIT_COMMIT"))
if _sha:
    _attrs["vcs.ref.head.revision"] = _sha
_resource = Resource.create(_attrs)

_tracer_provider = TracerProvider(resource=_resource)
_tracer_provider.add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(endpoint=f"{ENDPOINT}/v1/traces", headers=_HEADERS)))
trace.set_tracer_provider(_tracer_provider)

_meter_provider = MeterProvider(resource=_resource, metric_readers=[
    PeriodicExportingMetricReader(OTLPMetricExporter(
        endpoint=f"{ENDPOINT}/v1/metrics", headers=_HEADERS))])
metrics.set_meter_provider(_meter_provider)

_logger_provider = LoggerProvider(resource=_resource)
_logger_provider.add_log_record_processor(BatchLogRecordProcessor(
    OTLPLogExporter(endpoint=f"{ENDPOINT}/v1/logs", headers=_HEADERS)))
set_logger_provider(_logger_provider)
# Root-logger handler: app + library INFO logs ship via OTLP; console
# output is untouched (no stream handler added here).
logging.getLogger().addHandler(
    LoggingHandler(level=logging.INFO, logger_provider=_logger_provider))
logging.getLogger().setLevel(logging.INFO)
LoggingInstrumentor().instrument(set_logging_format=False)


def shutdown() -> None:
    """Flush all three signals — vital for a short-lived CLI."""
    _tracer_provider.shutdown()
    _logger_provider.shutdown()
    _meter_provider.shutdown()


atexit.register(shutdown)
