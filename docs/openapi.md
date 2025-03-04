# OpenAPI Documentation

The Local Operator API provides OpenAPI documentation that can be used to understand the API endpoints, request/response schemas, and to generate client code.

## Accessing OpenAPI Documentation

When the server is running, you can access the OpenAPI documentation in several ways:

1. **Swagger UI**: Visit `/docs` in your browser to access the interactive Swagger UI documentation.
   - Example: `http://localhost:8000/docs`

2. **ReDoc**: Visit `/redoc` in your browser to access the ReDoc documentation.
   - Example: `http://localhost:8000/redoc`

3. **Raw OpenAPI Schema**: Access the raw OpenAPI schema in JSON format at `/openapi.json`.
   - Example: `http://localhost:8000/openapi.json`

## Generating OpenAPI Schema

You can generate the OpenAPI schema without starting the server using the provided utility script:

```bash
# Generate the schema to the default location (~/.local-operator/docs/openapi.json)
python -m local_operator.server.generate_openapi

# Generate the schema to a specific location
python -m local_operator.server.generate_openapi -o /path/to/openapi.json
```

Alternatively, you can use the server with the `--generate-openapi` flag:

```bash
# Generate the schema to the default location
python -m local_operator.server.app --generate-openapi

# Generate the schema to a specific location
python -m local_operator.server.app --generate-openapi --openapi-output /path/to/openapi.json
```

## Using the OpenAPI Schema

The generated OpenAPI schema can be used with various tools:

1. **Client Code Generation**: Generate client code in various languages using tools like [OpenAPI Generator](https://openapi-generator.tech/).

   ```bash
   # Example: Generate a Python client
   openapi-generator generate -i ~/.local-operator/openapi.json -g python -o ./client
   ```

2. **API Testing**: Use tools like [Postman](https://www.postman.com/) or [Insomnia](https://insomnia.rest/) to import the OpenAPI schema and test the API endpoints.

3. **Documentation**: Use the schema to generate custom documentation or integrate with API documentation platforms.

## API Endpoints

The Local Operator API provides the following endpoint groups:

- **Health**: Health check endpoints
- **Chat**: Chat generation endpoints
- **Agents**: Agent management endpoints
- **Jobs**: Job management endpoints

For detailed information about each endpoint, including request/response schemas and examples, refer to the Swagger UI or ReDoc documentation.
