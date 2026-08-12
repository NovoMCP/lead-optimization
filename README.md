# Lead Optimization Service

Molecule generation service for the NovoMCP engine — scaffold hopping and property-directed optimization using RDKit.

ADMET and synthetic accessibility scoring are handled separately by the `addie-models` and `chem-props` services.

## Features

- **Scaffold Hopping**: Generate structural variants by swapping ring systems (benzene/pyridine, cyclohexane/piperidine, etc.)
- **Property-Directed Optimization**: Rank scaffold-hopped variants against target property values
- **Batch Lead Optimization**: Process multiple molecules and return top candidates by QED score

## API Endpoints

- `GET /health` - Health check
- `GET /about` - Service information
- `POST /scaffold-hop` - Generate scaffold variations for a single molecule
- `POST /optimize` - Optimize a molecule via scaffold hopping or property targeting
- `POST /optimize-leads` - Batch optimization across multiple molecules
- `GET /jobs/{job_id}/status` - Job status (runs synchronously in v3)

All `POST` endpoints require an `X-Api-Key` header.

## Configuration

| Variable | Description |
|---|---|
| `PORT` | Service port (default: `8023`) |
| `LEAD_OPT_API_KEY` | Required API key for authentication |

## Deployment

The service is a single stateless container.

```bash
# Pull and run the published image
docker run -p 8023:8023 ghcr.io/novomcp/lead-optimization:latest

# Or build from source
docker build -t lead-optimization .
docker run -p 8023:8023 lead-optimization
```

Set `LEAD_OPT_API_KEY` to require an `X-Api-Key` header; leave it unset for local development. Point the NovoMCP engine at this service by setting `LEAD_OPTIMIZATION_URL` to its URL.

## License

Licensed under the Apache License 2.0 (see `LICENSE`). Molecule handling uses RDKit (BSD-3-Clause).
