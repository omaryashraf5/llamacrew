"""Retry utilities for LlamaCrew."""

import logging
import time
from functools import wraps
from typing import Any, Callable, Optional, Type, Tuple

logger = logging.getLogger(__name__)


class RetryError(Exception):
    """Raised when retry attempts are exhausted."""

    pass


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""

    pass


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback called on each retry attempt

    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def flaky_function():
            # Your code here
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 1
            current_delay = delay

            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise RetryError(f"Failed after {max_attempts} attempts: {str(e)}") from e

                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1

        return wrapper

    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by temporarily blocking calls
    to a failing service.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Failure threshold exceeded, calls blocked
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                )

        try:
            result = func(*args, **kwargs)

            # Success - reset if in HALF_OPEN
            if self.state == "HALF_OPEN":
                self._reset()
                logger.info(f"Circuit breaker reset to CLOSED")

            return result

        except self.expected_exception as e:
            self._record_failure()
            raise

    def _record_failure(self) -> None:
        """Record a failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        if exc_type is not None and issubclass(exc_type, self.expected_exception):
            self._record_failure()
        return False
