"""Makefile shim: python -m eval.run → eval.loop_eval.main."""

import asyncio

from eval.loop_eval import main

if __name__ == "__main__":
    asyncio.run(main())
