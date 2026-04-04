import asyncio
import httpx


async def main():
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(
            "http://localhost:8000/predict",
            json={"video_path": "examples/earth.mp4"},
        )
        print(response.json())


asyncio.run(main())
