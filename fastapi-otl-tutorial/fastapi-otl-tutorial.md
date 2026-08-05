# Deploy OpenTelemetry Observability Stack on UbiOps with Docker and Services

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/fastapi-otl-tutorial/fastapi-otl-tutorial/){ .md-button .md-button--primary } [View source code :fontawesome-brands-github:](https://github.com/UbiOps/tutorials/blob/master/fastapi-otl-tutorial/fastapi-otl-tutorial/fastapi-otl-tutorial.ipynb){ .md-button }.

In this tutorial, we'll deploy a complete observability stack on UbiOps using OpenTelemetry. We'll create a FastAPI application that sends telemetry data to an OpenTelemetry Collector, which aggregates metrics and exposes them via Prometheus.

## What are UbiOps Services?

[UbiOps Services](https://ubiops.com/docs/services/) let you expose specific ports from your deployments through custom HTTP endpoints. Unlike standard UbiOps deployment endpoints, Services enable direct HTTP access to applications running inside your containers. This is essential for exposing endpoints like Prometheus metrics or API servers.

In this tutorial, we'll use Services to expose both the FastAPI application and the Prometheus metrics endpoint from the OpenTelemetry Collector.

## What is OpenTelemetry?

[OpenTelemetry](https://opentelemetry.io/) is an observability framework for cloud-native software. It provides a single set of APIs, libraries, and agents to capture distributed traces, metrics, and logs from your applications.

## What is FastAPI?

[FastAPI](https://fastapi.tiangolo.com/) is a modern, high-performance Python web framework for building APIs. It's built on standard Python type hints and provides automatic API documentation, data validation, and excellent performance. FastAPI is ideal for creating microservices and is widely used in production environments for its speed and developer experience.

## Tutorial Overview

We will:

- Build Docker images for the OpenTelemetry Collector and FastAPI application

- Create [custom environments](https://ubiops.com/docs/environments/#create-a-custom-environment) with these images

- Create [deployments](https://ubiops.com/docs/deployments/) that run both services

- Expose endpoints via [UbiOps Services](https://ubiops.com/docs/services/)

- Configure authentication settings (disabled for Prometheus, enabled for FastAPI)

- Test the observability stack with load testing

**Prerequisites:**

- [Docker Engine](https://docs.docker.com/engine/) or [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed

- UbiOps account with Services enabled and Bring-your-own-docker-image functionality enabled.

## Architecture Overview

We'll deploy two components:

1. **OpenTelemetry Collector** - Receives telemetry data via OTLP HTTP (port 4318) and exports aggregated metrics via Prometheus format (port 8889)
2. **FastAPI Application** - Simple timer service that sends custom metrics and HTTP instrumentation to the collector

### Data Flow

![Data Flow](https://storage.googleapis.com/ubiops/public-docs-media/images/data-flow.png)

### Port Mapping

| Port | Protocol | Purpose | Authentication |
|------|----------|---------|----------------|
| 4318 | OTLP HTTP | Write telemetry | **Disabled** |
| 8889 | Prometheus | Read metrics | **Disabled** |
| 8080 | FastAPI | Timer API | Enabled |

### Visualizing the data

The Prometheus endpoint we will expose can be connected to [Grafana](https://grafana.com/) for metric visualizations and dashboards. Since UbiOps Services provide external HTTPS endpoints, you have two options:

**Option 1: Client-side Grafana** - Run Grafana locally or on your own infrastructure and point it to the UbiOps Service URL (`https://xxx.services.ubiops.com/metrics`). This is the simplest approach for getting started.

**Option 2: Grafana on UbiOps** - Deploy Grafana as another UbiOps deployment and expose it via Services. This keeps everything in one platform and allows team access through UbiOps permissions.

**Some Resources to get you started:**
- [Grafana Getting Started](https://grafana.com/docs/grafana/latest/fundamentals/getting-started/) - Install and configure Grafana
- [Prometheus Data Source](https://grafana.com/docs/grafana/latest/datasources/prometheus/) - Connect Grafana to Prometheus
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/) - Pre-built dashboard templates
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/) - Learn more about observability

## 1. Set up connection with UbiOps API client

Install the [UbiOps Python Client Library](https://ubiops.com/docs/python_client_library/) and initialize the connection.


```python
!pip install -qU ubiops requests
```

Initialize variables for the UbiOps deployment. See [here](https://ubiops.com/docs/organizations/service-users/) to learn how to get your API token.


```python
# Initialize variables
API_TOKEN = "<INSERT API TOKEN WITH PROJECT EDITOR RIGHTS>"
PROJECT_NAME = "<INSERT YOUR PROJECT NAME>"
API_HOST_URL = "https://api.ubiops.com/v2.1"  

# Deployment names
COLLECTOR_DEPLOYMENT = "otel-collector"
FASTAPI_DEPLOYMENT = "fastapi-timer-service"
DEPLOYMENT_VERSION = "v1"

# Environment names
COLLECTOR_ENV = "otel-collector-env"
FASTAPI_ENV = "fastapi-otel-env"

# Service names
COLLECTOR_OTLP_SERVICE = "otel-http"         # Port 4318 - OTLP receiver
COLLECTOR_PROM_SERVICE = "otel-prometheus"   # Port 8889 - Prometheus exporter
FASTAPI_SERVICE = "fastapi-timer"            # Port 8080 - FastAPI app

print(f"Collector deployment: {COLLECTOR_DEPLOYMENT}")
print(f"FastAPI deployment: {FASTAPI_DEPLOYMENT}")
print(f"Services to create: {COLLECTOR_OTLP_SERVICE}, {COLLECTOR_PROM_SERVICE}, {FASTAPI_SERVICE}")
```


```python
# Initialize UbiOps client
import ubiops

configuration = ubiops.Configuration(host=API_HOST_URL)
configuration.api_key["Authorization"] = API_TOKEN

client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)

# Test connection
api.service_status()
```

### Alternative Deployment Options

This tutorial uses custom Docker images via UbiOps [bring your own Docker feature](https://ubiops.com/docs/deployments/docker-support/), which provides maximum flexibility for deploying pre-configured applications. However, if you don't have access to custom Docker environments, you have alternatives:

**For the FastAPI application**, you can use [UbiOps managed environments](https://ubiops.com/docs/environments/) with a [deployment package](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/). Instead of building a Docker image, create a deployment package with your `requirements.txt` and a `deployment.py` file that starts the FastAPI server. This approach works with standard Python environments and doesn't require Docker access.

**For the OpenTelemetry Collector**, the configuration is more complex and benefits from the containerized approach. If custom Docker environments aren't available, consider hosting the collector externally (on your own infrastructure, AWS, or Google Cloud) and pointing your UbiOps deployments to that external endpoint. The collector can receive telemetry from anywhere via HTTPS.

## 2. Build Docker images

We'll build two Docker images: one for the OpenTelemetry Collector and one for the FastAPI application.

### 2.1 Create OpenTelemetry Collector image

The collector receives telemetry via OTLP HTTP protocol and exports metrics in Prometheus format.


```python
%%writefile otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 10s
    send_batch_size: 1024

exporters:
  debug:
    verbosity: detailed
  
  prometheus:
    endpoint: "0.0.0.0:8889"
    namespace: ""
    send_timestamps: true
    metric_expiration: 5m
    enable_open_metrics: true

extensions:
  health_check:
    endpoint: 0.0.0.0:13133

service:
  extensions: [health_check]
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug, prometheus]
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
```


```python
%%writefile Dockerfile.collector
FROM otel/opentelemetry-collector:0.142.0

# Copy collector configuration
COPY otel-collector-config.yaml /etc/otel-collector-config.yaml

# Expose ports
EXPOSE 4318 8889 13133

# Run collector
ENTRYPOINT ["/otelcol"]
CMD ["--config=/etc/otel-collector-config.yaml"]
```


```python
# Build collector image
!docker build --platform linux/amd64 -f Dockerfile.collector -t otel-collector-ubiops .
```


```python
# Save and compress collector image
!docker save otel-collector-ubiops -o otel-collector.tar
!gzip otel-collector.tar
```

### 2.2 Create FastAPI application image

The FastAPI app is a simple timer service that sleeps for a specified number of seconds. It automatically sends HTTP instrumentation and custom metrics to the collector, including request counts, response times, and sleep durations.


```python
%%writefile app.py
from fastapi import FastAPI, Query
import time
import os
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Environment variables
# OTEL_COLLECTOR_ENDPOINT will be set via UbiOps environment variable
# to the external Service URL (https://<service.id>.services.ubiops.com)
OTEL_COLLECTOR_ENDPOINT = os.getenv(
    "OTEL_COLLECTOR_ENDPOINT"
)
SERVICE_NAME = os.getenv("SERVICE_NAME", "fastapi-timer-service")
INSTANCE_ID = os.getenv("UBIOPS_DEPLOYMENT_INSTANCE_ID", "local-instance")[:8]

# Configure OpenTelemetry
resource = Resource(attributes={
    "service.name": SERVICE_NAME,
    "deployment.environment": "development",
    "instance_id": INSTANCE_ID
})

# Setup metrics exporter
metric_exporter = OTLPMetricExporter(
    endpoint=f"{OTEL_COLLECTOR_ENDPOINT}/v1/metrics",
    timeout=10
)

metric_reader = PeriodicExportingMetricReader(
    exporter=metric_exporter,
    export_interval_millis=5000
)

meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[metric_reader]
)
metrics.set_meter_provider(meter_provider)

# Create custom metrics
meter = metrics.get_meter("app")
health_check_counter = meter.create_counter(
    "health_checks_total",
    description="Total number of health check requests"
)
timer_counter = meter.create_counter(
    "timer_requests_total",
    description="Total number of timer requests"
)
sleep_histogram = meter.create_histogram(
    "sleep_duration_seconds",
    description="Duration of sleep in timer endpoint"
)

# Create FastAPI app
app = FastAPI(title="Timer Service with OpenTelemetry")

# Instrument FastAPI (automatic HTTP metrics)
FastAPIInstrumentor.instrument_app(app)

@app.get("/health")
def health():
    health_check_counter.add(1, {"instance_id": INSTANCE_ID})
    return {"status": "healthy", "instance_id": INSTANCE_ID}

@app.get("/timer")
def timer(seconds: int = Query(default=1, ge=1, le=60)):
    timer_counter.add(1, {"instance_id": INSTANCE_ID, "endpoint": "/timer"})
    time.sleep(seconds)
    sleep_histogram.record(seconds, {"instance_id": INSTANCE_ID})
    return {
        "message": f"Slept for {seconds} seconds",
        "instance_id": INSTANCE_ID,
        "seconds": seconds
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```


```python
%%writefile requirements.txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
opentelemetry-api==1.29.0
opentelemetry-sdk==1.29.0
opentelemetry-exporter-otlp-proto-http==1.29.0
opentelemetry-instrumentation-fastapi==0.50b0
```


```python
%%writefile Dockerfile.fastapi
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "app.py"]
```


```python
# Build FastAPI image
!docker build --platform linux/amd64 -f Dockerfile.fastapi -t fastapi-otel-ubiops .
```


```python
# Save and compress FastAPI image
!docker save fastapi-otel-ubiops -o fastapi-otel.tar
!gzip fastapi-otel.tar
```

## 3. Create custom environments in UbiOps

UbiOps has [Bring you own Docker Image feature](https://ubiops.com/docs/environments/#bring-your-own-docker-image) which allows you to upload Docker images to UbiOps. We'll create two custom environments for our observability stack and upload our Docker Images as revisions.


```python
# Create collector environment
collector_env = api.environments_create(
    project_name=PROJECT_NAME,
    data=ubiops.EnvironmentCreate(
        name=COLLECTOR_ENV,
        display_name="OpenTelemetry Collector",
        base_environment="ubuntu22-04-python3-11",
        description="OpenTelemetry Collector with Prometheus exporter"
    )
)

print(f"Created environment: {collector_env.name}")
```


```python
# Upload collector Docker image
upload_response = api.environment_revisions_file_upload(
    project_name=PROJECT_NAME,
    environment_name=COLLECTOR_ENV,
    file="otel-collector.tar.gz"
)

print(f"Uploaded collector image, revision: {upload_response.revision}")

ubiops.utils.wait_for_environment(
      api, PROJECT_NAME, COLLECTOR_ENV,
      timeout=1800, quiet=False, stream_logs=False
  )
```


```python
# Create FastAPI environment
fastapi_env = api.environments_create(
    project_name=PROJECT_NAME,
    data=ubiops.EnvironmentCreate(
        name=FASTAPI_ENV,
        display_name="FastAPI OpenTelemetry",
        base_environment="ubuntu22-04-python3-11",
        description="FastAPI with OpenTelemetry instrumentation"
    )
)

print(f"Created environment: {fastapi_env.name}")
```


```python
# Upload FastAPI Docker image
upload_response = api.environment_revisions_file_upload(
    project_name=PROJECT_NAME,
    environment_name=FASTAPI_ENV,
    file="fastapi-otel.tar.gz"
)

print(f"Uploaded FastAPI image, revision: {upload_response.revision}")

ubiops.utils.wait_for_environment(
      api, PROJECT_NAME, FASTAPI_ENV,
      timeout=1800, quiet=False, stream_logs=False
)
```

## 4. Create deployments

We'll create two [deployments](https://ubiops.com/docs/deployments/): one for the collector and one for FastAPI.

### 4.1 Create OpenTelemetry Collector deployment


```python
# Create collector deployment
collector_deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data=ubiops.DeploymentCreate(
        name=COLLECTOR_DEPLOYMENT,
        description="OpenTelemetry Collector for metrics aggregation",
        supports_request_format=False
    )
)

print(f"Created deployment: {collector_deployment.name}")
```


```python
# Create collector deployment version
collector_version = api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=COLLECTOR_DEPLOYMENT,
    data=ubiops.DeploymentVersionCreate(
        version=DEPLOYMENT_VERSION,
        environment=COLLECTOR_ENV,
        instance_type_group_name="512 MB + 0.125 vCPU",
        minimum_instances=1,
        request_retention_mode="Full",
    )
)

print(f"Created version: {collector_version.version}")
print("Waiting for deployment to become available...")

ubiops.utils.wait_for_deployment_version(
    api, PROJECT_NAME, COLLECTOR_DEPLOYMENT, DEPLOYMENT_VERSION, revision_id=collector_version.revision,
    timeout=1800, quiet=False, stream_logs=False
)
```

### 4.2 Create FastAPI deployment


```python
# Create FastAPI deployment
fastapi_deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data=ubiops.DeploymentCreate(
        name=FASTAPI_DEPLOYMENT,
        description="FastAPI timer service with OpenTelemetry",
        supports_request_format=False
    )
)

print(f"Created deployment: {fastapi_deployment.name}")
```


```python
# Create FastAPI deployment version
fastapi_version = api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=FASTAPI_DEPLOYMENT,
    data=ubiops.DeploymentVersionCreate(
        version=DEPLOYMENT_VERSION,
        environment=FASTAPI_ENV,
        instance_type_group_name="512 MB + 0.125 vCPU",
        minimum_instances=3,
        request_retention_mode="Full",
        environment_variables=[
            {
                "name": "SERVICE_NAME",
                "value": "fastapi-timer-service",
                "secret": False
            }
            # OTEL_COLLECTOR_ENDPOINT will be added after creating OTLP service
        ]
    )
)

print(f"Created version: {fastapi_version.version}")
print("Waiting for deployment to become available...")

ubiops.utils.wait_for_deployment_version(
    api, PROJECT_NAME, FASTAPI_DEPLOYMENT, DEPLOYMENT_VERSION, revision_id=fastapi_version.revision,
    timeout=1800, quiet=False, stream_logs=False
)
```

## 5. Create UbiOps Services

[Services](https://ubiops.com/docs/python_client_library/Services/) expose specific ports from deployments through external HTTPS endpoints. 
We'll create three services to expose different ports from our two deployments:

1. OTLP HTTP endpoint (port 4318) - Collector endpoint for receiving telemetry data
2. Prometheus metrics (port 8889) - Collector endpoint for exporting aggregated metrics
3. FastAPI endpoint (port 8080) - Application endpoint for the timer API

This design allows us to expose multiple endpoints from the same deployment. 
The OpenTelemetry Collector deployment exposes two services on different ports: 
one for writing telemetry data (OTLP) and another for reading metrics (Prometheus).
This separation of concerns follows observability best practices where data 
ingestion and data export happen through different interfaces.

We disable authentication on the OTLP and Prometheus endpoints for simplicity. 
The OTLP endpoint is used for internal communication between FastAPI instances 
and the collector, while the Prometheus endpoint provides read-only access to 
aggregated metrics. The FastAPI endpoint keeps authentication enabled as it's 
the main user-facing API.


```python
# Create OTLP HTTP service
api.services_create(
    project_name=PROJECT_NAME,
    data=ubiops.ServiceCreate(
        name=COLLECTOR_OTLP_SERVICE,
        deployment=COLLECTOR_DEPLOYMENT,
        version=DEPLOYMENT_VERSION,
        port=4318,
        enabled_authentication=False
    )
)

# Get service details to retrieve endpoint URL
otlp_service = api.services_get(
    project_name=PROJECT_NAME,
    service_name=COLLECTOR_OTLP_SERVICE
)

print(f"Created OTLP service: {otlp_service.endpoint}")

# Save OTLP endpoint for FastAPI configuration
OTLP_ENDPOINT = otlp_service.endpoint
```


```python
# Create Prometheus service
api.services_create(
    project_name=PROJECT_NAME,
    data=ubiops.ServiceCreate(
        name=COLLECTOR_PROM_SERVICE,
        deployment=COLLECTOR_DEPLOYMENT,
        version=DEPLOYMENT_VERSION,
        port=8889,
        enabled_authentication=False
    )
)

# Get service details to retrieve endpoint URL
prom_service = api.services_get(
    project_name=PROJECT_NAME,
    service_name=COLLECTOR_PROM_SERVICE
)

print(f"Created Prometheus service: {prom_service.endpoint}")
print(f"  Access metrics at: {prom_service.endpoint}/metrics")
```


```python
# Create FastAPI service (authentication ENABLED - API access)
api.services_create(
    project_name=PROJECT_NAME,
    data=ubiops.ServiceCreate(
        name=FASTAPI_SERVICE,
        deployment=FASTAPI_DEPLOYMENT,
        version=DEPLOYMENT_VERSION,
        port=8080,
        enabled_authentication=True
    )
)

# Get service details to retrieve endpoint URL
fastapi_service = api.services_get(
    project_name=PROJECT_NAME,
    service_name=FASTAPI_SERVICE
)

print(f"Created FastAPI service: {fastapi_service.endpoint}")
print(f"  Authentication: ENABLED (requires token)")
```


```python
# Save service URLs for later use
PROMETHEUS_URL = prom_service.endpoint
FASTAPI_URL = fastapi_service.endpoint

print("\n" + "="*70)
print("Service URLs:")
print("="*70)
print(f"Prometheus metrics: {PROMETHEUS_URL}/metrics (NO AUTH)")
print(f"FastAPI timer:      {FASTAPI_URL}/timer?seconds=5 (AUTH REQUIRED)")
print(f"FastAPI health:     {FASTAPI_URL}/health (AUTH REQUIRED)")
print("="*70)
```

## 5.1 Update FastAPI deployment with OTLP endpoint

Now that we have the OTLP service URL, we need to update the FastAPI deployment to use it. We'll use the external Service URL instead of internal Kubernetes DNS.


```python
# Update FastAPI deployment with OTLP collector URL
api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=FASTAPI_DEPLOYMENT,
    version=DEPLOYMENT_VERSION,
    data=ubiops.EnvironmentVariableCreate(
        name="OTEL_COLLECTOR_ENDPOINT",
        value=OTLP_ENDPOINT,
        secret=False
    )
)

print(f"Updated FastAPI with OTLP endpoint: {OTLP_ENDPOINT}")
print("\nNote: Using external Service URL instead of internal DNS")
print("This allows telemetry to flow through UbiOps Services infrastructure")
```

## 6. Test the observability stack

### 6.1 Test Prometheus metrics endpoint (no authentication required)


```python
import requests

# Fetch Prometheus metrics (no authentication needed)
response = requests.get(f"{PROMETHEUS_URL}/metrics")

if response.status_code == 200:
    # Show first 50 lines
    lines = response.text.split('\n')[:50]
    print("Prometheus metrics (first 50 lines):")
    print('\n'.join(lines))
else:
    print(f"Error: {response.status_code}")
```

### 6.2 Test FastAPI endpoints (authentication required)


```python
# Test health endpoint
response = requests.get(
    f"{FASTAPI_URL}/health",
    headers={"Authorization": f"Token {API_TOKEN}"}
)

print("Health check response:")
print(response.json())
```


```python
# Test timer endpoint
response = requests.get(
    f"{FASTAPI_URL}/timer?seconds=3",
    headers={"Authorization": f"Token {API_TOKEN}"}
)

print("Timer response:")
print(response.json())
```

## 7. Load testing

Send 100 concurrent requests to see load distribution across instances.


```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

def send_request(request_id):
    """Send a single timer request"""
    try:
        response = requests.get(
            f"{FASTAPI_URL}/timer?seconds=1",
            headers={"Authorization": f"Token {API_TOKEN}"},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'instance_id': data.get('instance_id', 'unknown')[:8],
                'request_id': request_id
            }
        else:
            return {'success': False, 'request_id': request_id}
    except Exception as e:
        return {'success': False, 'error': str(e), 'request_id': request_id}

# Send 100 requests with 20 concurrent threads
NUM_REQUESTS = 100
MAX_WORKERS = 20

print(f"Sending {NUM_REQUESTS} requests with {MAX_WORKERS} concurrent threads...")

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(send_request, i) for i in range(NUM_REQUESTS)]
    
    completed = 0
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        completed += 1
        
        if completed % 20 == 0:
            print(f"  Progress: {completed}/{NUM_REQUESTS}")

# Analyze results
successful = [r for r in results if r.get('success')]
failed = [r for r in results if not r.get('success')]

instance_distribution = Counter([r['instance_id'] for r in successful])

print(f"\n{'='*70}")
print(f"Results:")
print(f"Successful: {len(successful)}/{NUM_REQUESTS}")
print(f"Failed: {len(failed)}/{NUM_REQUESTS}")

print(f"\nInstance Distribution:")
for instance, count in instance_distribution.most_common():
    percentage = (count / len(successful)) * 100
    bar = "█" * int(percentage / 2)
    print(f"  {instance}: {count:3d} requests ({percentage:5.1f}%) {bar}")

print(f"{'='*70}")
```

### View metrics after load test


```python
# Wait for metrics to export
print("Waiting 15 seconds for metrics to export...")
import time
time.sleep(15)

# Fetch updated metrics
response = requests.get(f"{PROMETHEUS_URL}/metrics")

# Parse timer_requests_total metric
print("\nTimer requests per instance:")
for line in response.text.split('\n'):
    if 'timer_requests_total{' in line and not line.startswith('#'):
        print(f"  {line}")
```

## 8. Cleanup

Scale down deployments to avoid extra charges.


```python
raise SystemExit("Prevented running all cells to avoid scaling down before completing all tests. Execute manually to scale down.")
```


```python
# Scale down FastAPI deployment
api.deployment_versions_update(
    project_name=PROJECT_NAME,
    deployment_name=FASTAPI_DEPLOYMENT,
    version=DEPLOYMENT_VERSION,
    data=ubiops.DeploymentVersionUpdate(minimum_instances=0)
)

# Scale down Collector deployment
api.deployment_versions_update(
    project_name=PROJECT_NAME,
    deployment_name=COLLECTOR_DEPLOYMENT,
    version=DEPLOYMENT_VERSION,
    data=ubiops.DeploymentVersionUpdate(minimum_instances=0)
)

print("Scaled deployments to 0 instances")
```


```python
# Close API client
client.close()
print("Closed UbiOps connection")
```

You've successfully deployed a complete OpenTelemetry observability stack on UbiOps!

Reach out to [UbiOps support](https://www.support.ubiops.com) if you have any questions.
