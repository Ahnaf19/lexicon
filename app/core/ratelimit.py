"""Token-bucket rate limiters for external APIs (28 RPM for Groq, under the 30 RPM cap)."""

from aiolimiter import AsyncLimiter

groq_limiter = AsyncLimiter(28, 60)
