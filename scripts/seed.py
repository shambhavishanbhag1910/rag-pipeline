from __future__ import annotations

import argparse
import asyncio

from app.core.config import get_settings
from app.db import AsyncSessionLocal
from app.services.synthetic import seed_synthetic_data


async def run(tenant_id: str, reset: bool) -> None:
    settings = get_settings()
    async with AsyncSessionLocal() as session:
        results = await seed_synthetic_data(
            session,
            base_path=settings.synthetic_data_path,
            tenant_id=tenant_id,
            reset=reset,
        )
    for result in results:
        print(f"{result.status:9} {result.source_id:28} chunks={result.chunks}")
    print(f"Seeded {len(results)} documents for tenant={tenant_id!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed bundled synthetic enterprise documents")
    parser.add_argument("--tenant-id", default="demo")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.tenant_id, args.reset))


if __name__ == "__main__":
    main()
