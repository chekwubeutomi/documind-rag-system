"""
Main FastAPI application entry point.

This module defines the FastAPI application and all middleware/lifecycle hooks.
It's the heart of the REST API - where requests come in and responses go out.

Application Architecture:
```
Client (Browser, Python, cURL)
    ↓
HTTP Request
    ↓
FastAPI (app instance)
    ├─ Middleware layers (process request)
    ├─ Route matching (find correct endpoint)
    ├─ Endpoint handler (execute business logic)
    └─ Middleware layers (process response)
    ↓
HTTP Response
    ↓
Client receives response
```

FastAPI Features Used:
- APIRouter: Organizes endpoints into logical groups (/v1/*, etc)
- Middleware: Pre-process requests, post-process responses
- Exception handlers: Catch errors and return JSON responses
- Lifespan: Run code on app startup/shutdown
- OpenAPI/Swagger: Auto-generated API documentation at /docs

Middleware Pipeline (execution order):
1. CORS middleware: Handle Cross-Origin requests from browsers
2. Request logging: Log incoming requests (method, path)
3. Application logic (route handlers)
4. Response logging: Log outgoing responses (status code)
5. Exception handling: Catch and format errors
6. Response returns to client

Startup Flow (when app starts):
1. setup_logging(): Configure logger with rotating file handler
2. FastAPI.__init__(): Create app instance
3. add_middleware(): Add CORS, logging
4. include_router(): Register /v1 endpoints
5. Lifespan.startup: Called, runs initialize_services()
   - DocumentProcessor ready
   - EmbeddingService model loaded
   - LLMService client initialized
   - VectorDBService connected
   - Qdrant collection created
6. App ready to accept requests!

Shutdown Flow (when app stops):
1. SIGTERM signal received (from container, Kubernetes, etc)
2. Lifespan.shutdown: Run cleanup (log application stopping)
3. FastAPI stops accepting requests
4. All services gracefully close connections
5. App exits

Error Handling:
- Global exception handler: Catches all unhandled exceptions
- Returns JSON error responses (never HTML error pages)
- DEBUG mode: includes full error details
- Production mode: hides sensitive error info

Prometheus Metrics (optional):
- Tracks HTTP request metrics (latency, counts, errors)
- Exposes metrics at /metrics endpoint
- Integrates with Grafana dashboards
- Requires: pip install prometheus-fastapi-instrumentator
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.api.v1 import endpoints

# ======================================================================
# Application Startup: Configure Logging
# ======================================================================
# This must be done BEFORE FastAPI app creation so we can log startup messages
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup/shutdown events.
    
    In FastAPI, lifespan replaces the old startup/shutdown events.
    It uses a context manager (yield) pattern:
    - Before yield: Startup code (runs once when app starts)
    - After yield: Shutdown code (runs once when app stops)
    
    Startup Sequence:
    1. Log that app is starting
    2. Call endpoints.initialize_services() which:
       - Creates DocumentProcessor
       - Loads embedding model (downloads if first time)
       - Initializes LLM client
       - Connects to vector database
       - Creates collection for vectors
    3. All services ready for incoming requests
    
    Shutdown Sequence:
    1. Receive shutdown signal (SIGTERM from container)
    2. Log that app is shutting down
    3. Services close connections gracefully
    4. App process exits
    
    Example Lifespan Flow:
    
    App Start
        ↓
    lifespan() called
        ↓
    Before yield: startup code runs ← You are here
        ↓
    yield
        ↓
    [App serving requests for hours/days]
        ↓
    Docker container stops / Kubernetes kills pod
        ↓
    After yield: shutdown code runs
        ↓
    App exits
    """
    # ======================================================================
    # Startup Code - Runs once when app starts
    # ======================================================================
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize all services (LLM, embeddings, vector DB, etc)
    # This can take 10-30 seconds if downloading embedding model
    endpoints.initialize_services()
    
    # Yield control: app is now ready to accept requests
    # The app will serve requests while yielding
    yield
    
    # ======================================================================
    # Shutdown Code - Runs once when app stops
    # ======================================================================
    logger.info("Shutting down application")
    # Could add cleanup here: close database connections, flush metrics, etc


# ======================================================================
# FastAPI Application Setup
# ======================================================================
# Create the FastAPI application instance
# This is THE app object that receives all HTTP requests
app = FastAPI(
    title=settings.APP_NAME,  # "DocMind" - shown in /docs page
    description="Document Retrieval-Augmented Generation System",  # API description
    version=settings.APP_VERSION,  # e.g., "0.1.0"
    debug=settings.DEBUG,  # Enable/disable debug mode
    lifespan=lifespan  # Register our startup/shutdown events
)


# ======================================================================
# MIDDLEWARE CONFIGURATION
# ======================================================================
# Middleware processes requests BEFORE they reach endpoint handlers
# And processes responses AFTER endpoint handlers return

# CORS Middleware - Allow requests from any origin
# Without this, browser-based clients get CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow from any origin (can restrict in production)
    allow_credentials=True,  # Allow cookies/credentials in requests
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, DELETE, etc)
    allow_headers=["*"],  # Allow all headers (authorization, content-type, etc)
)


# Request Logging Middleware - Log all incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log HTTP requests and responses.
    
    This middleware runs for every request.
    It logs:
    - Incoming: method, path (e.g., "POST /v1/query")
    - Outgoing: method, path, status code
    
    Flow:
    1. Request arrives: log it
    2. call_next(): Pass to next middleware/endpoint
    3. Response returned: log status code
    4. Return response to client
    
    Example Log Output:
        INFO POST /v1/query
        INFO POST /v1/query - 200
    
    Args:
        request: The incoming HTTP request
        call_next: Function to call next handler
    
    Returns:
        The response from the handler
    """
    # Log the incoming request
    logger.info(f"{request.method} {request.url.path}")
    
    # Process the request (call endpoint handler)
    # The response object is created here
    response = await call_next(request)
    
    # Log the response status code
    logger.info(f"{request.method} {request.url.path} - {response.status_code}")
    
    return response


# ======================================================================
# EXCEPTION HANDLERS
# ======================================================================
# Exception handlers catch errors and return JSON responses instead of HTML

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    
    If an endpoint raises an exception that isn't caught,
    this handler will catch it and return a JSON error response.
    
    Without this handler, FastAPI would return an HTML error page
    (not good for API clients).
    
    Args:
        request: The request that caused the error
        exc: The exception that was raised
    
    Returns:
        JSON response with 500 status and error details
        In DEBUG mode: includes full error message
        In PRODUCTION: hides error details for security
    
    Example Response:
    {
        "error": "Internal Server Error",
        "message": "division by zero" (only in DEBUG)
    }
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,  # 500 = Internal Server Error
        content={
            "error": "Internal Server Error",
            # In debug mode: show error details (helps developers)
            # In production: hide details (don't leak info to clients)
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# ======================================================================
# ROUTE DEFINITIONS
# ======================================================================
# These are the actual endpoints clients call

@app.get("/")
async def root():
    """
    Root endpoint / welcome page.
    
    Returns general information about the API.
    
    HTTP Method: GET (read-only)
    Path: /
    Auth: None required
    
    Returns:
        JSON with app name, version, and link to API docs
        
    Example Response:
    {
        "message": "Welcome to DocMind",
        "version": "0.1.0",
        "docs": "/docs"
    }
    """
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs"  # Link to interactive API documentation
    }


# Include v1 API routes
# This adds all endpoints from endpoints.router with /v1 prefix
# So @router.post("/query") becomes /v1/query
app.include_router(endpoints.router, prefix=settings.API_V1_STR)


# ======================================================================
# OPTIONAL: Prometheus Metrics Collection
# ======================================================================
# Prometheus collects metrics for monitoring and alerting
# Dashboard: Grafana (displays real-time graphs)
# Metrics: Request count, latency, errors

if settings.ENABLE_PROMETHEUS:
    try:
        # Import Prometheus integration
        from prometheus_fastapi_instrumentator import Instrumentator
        
        # Add Prometheus instrumentation
        # This wraps all endpoints to collect metrics
        Instrumentator().instrument(app).expose(app)
        logger.info("Prometheus metrics enabled at /metrics")
    except ImportError:
        # prometheus-fastapi-instrumentator not installed
        logger.warning("prometheus-fastapi-instrumentator not installed")
    except Exception as e:
        # Error setting up Prometheus (but don't crash app)
        logger.error(f"Error setting up Prometheus: {e}")


# ======================================================================
# Development Server
# ======================================================================
# This runs only when script is executed directly (not when imported)

if __name__ == "__main__":
    import uvicorn
    
    # Uvicorn is the ASGI server that actually runs FastAPI
    # (FastAPI itself is just a framework; uvicorn handles networking)
    uvicorn.run(
        "main:app",  # Path to app object
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,  # Listen on port 8000
        reload=settings.DEBUG,  # Auto-reload when code changes (debug only)
        log_level=settings.LOG_LEVEL.lower()  # INFO, DEBUG, WARNING, ERROR
    )
