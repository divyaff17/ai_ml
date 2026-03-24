import asyncio
from httpx import ASGITransport, AsyncClient
import main


async def run():
    print("Testing ASGITransport lifespan...")
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        print("Client is active. Checking app.state:")
        print("hasattr vision_model:", hasattr(main.app.state, "vision_model"))
        print("hasattr face_extractor:", hasattr(main.app.state, "face_extractor"))
        print("hasattr supabase:", hasattr(main.app.state, "supabase"))
        try:
            print("Sending GET /health...")
            r = await ac.get("/health")
            print("GET /health ->", r.status_code, r.text)

            print("Sending POST /detect/image...")
            with open("tests/test_api.py", "rb") as f:
                r = await ac.post(
                    "/detect/image",
                    files={"file": ("test.jpg", f, "image/jpeg")},
                    headers={"X-API-Key": "test-api-secret"},
                )
            print("POST /detect/image ->", r.status_code, r.text)
        except Exception as e:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run())
