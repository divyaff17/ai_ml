def run_model_check():
    print("--- MODEL LAYER CHECK ---")
    try:
        from models.vision_model import DeepfakeVisionModel, FaceExtractor
        from models.vision_model import fuse_scores
        from models.audio_model import DeepfakeAudioModel
        from PIL import Image
        import io

        model = DeepfakeVisionModel()
        img = Image.new("RGB", (224, 224), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = model.predict(buf.getvalue())
        assert "is_fake" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        print("Vision model OK:", result)

        fe = FaceExtractor()
        faces = fe.extract_from_image(buf.getvalue())
        print("Face extractor OK, faces found:", len(faces))

        score, is_fake = fuse_scores(0.9, 0.8, 0.7)
        assert isinstance(score, float)
        assert is_fake == True
        score2, is_fake2 = fuse_scores(0.1, 0.1, None)
        assert is_fake2 == False
        print("Fusion OK")

        audio = DeepfakeAudioModel()
        print("Audio model loaded OK")

        print("ALL MODEL CHECKS PASSED\n")
    except Exception as e:
        print("MODEL CHECK FAILED:", e)


def run_supabase_check():
    print("--- SUPABASE CONNECTION CHECK ---")
    try:
        from config import SUPABASE_URL, SUPABASE_KEY
        from supabase import create_client
        import uuid

        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        job_id = str(uuid.uuid4())

        res = (
            sb.table("detections")
            .insert(
                {
                    "job_id": job_id,
                    "media_type": "image",
                    "is_fake": False,
                    "confidence": 0.12,
                    "status": "completed",
                }
            )
            .execute()
        )
        assert res.data, "Insert failed"
        print("Insert OK")

        res2 = sb.table("detections").select("*").eq("job_id", job_id).execute()
        assert len(res2.data) == 1
        print("Fetch OK")

        sb.table("detections").delete().eq("job_id", job_id).execute()
        print("Cleanup OK")

        print("SUPABASE CONNECTION OK\n")
    except Exception as e:
        print("SUPABASE CHECK FAILED:", e)


def run_celery_check():
    print("--- CELERY + REDIS CHECK ---")
    try:
        from tasks.celery_app import celery

        result = celery.control.inspect().ping()
        print("Celery ping:", result)
        print("CELERY CHECK PASSED\n")
    except Exception as e:
        print("CELERY PING FAILED (Make sure redis and worker are running):", e)


def run_agent_check():
    print("--- AGENT CHECK ---")
    try:
        from unittest.mock import patch
        from langchain.tools import Tool
        from agents.investigation_agent import DeepfakeInvestigationAgent

        mock_tavily_tool_instance = Tool(
            name="tavily_search_results_json",
            description="mock",
            func=lambda *args, **kwargs: "mocked results",
        )

        with patch(
            "agents.investigation_agent.TavilySearchResults",
            return_value=mock_tavily_tool_instance,
        ) as mock_tavily, patch("agents.investigation_agent.ChatAnthropic") as mock_llm:

            mock_llm.return_value.invoke.return_value.content = """
            {
              "origin_found": false,
              "confidence_assessment": "likely_fake",
              "sources": [],
              "exif_flags": [],
              "summary": "Test summary."
            }"""

            agent = DeepfakeInvestigationAgent(confidence=0.95)
            result = agent.investigate("test_job_123")

            required_keys = [
                "origin_found",
                "confidence_assessment",
                "sources",
                "exif_flags",
                "summary",
            ]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"

            print("Agent output:", result)
            print("AGENT CHECK PASSED\n")
    except Exception as e:
        print("AGENT CHECK FAILED:", e)


if __name__ == "__main__":
    run_model_check()
    run_supabase_check()
    run_celery_check()
    run_agent_check()
