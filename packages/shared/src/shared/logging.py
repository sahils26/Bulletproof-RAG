"""Structured Logging Setup — powered by structlog.

This module configures structlog for the entire Bulletproof RAG system.
All components should use `get_logger()` to obtain a logger instance.

Why structlog?
- JSON output makes logs machine-parseable (great for Phase 3 observability)
- Bound context (logger.bind(component="llm")) carries metadata automatically
- Human-readable in dev mode, JSON in production
"""

import structlog


def configure_logging(json_output: bool = False) -> None:
    """Configure structlog globally.

    Args:
        json_output: If True, output structured JSON logs.
                     If False, output human-readable colored logs.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound to a specific component.

    Args:
        component: Name of the component (e.g., "llm", "ingestion",
                   "retrieval").

    Returns:
        A structlog BoundLogger with the component name pre-bound.
    """
    return structlog.get_logger(component=component)
