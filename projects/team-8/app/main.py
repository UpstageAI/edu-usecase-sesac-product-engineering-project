from fastapi import FastAPI

from app.agents.orchestrator.orchestrator_graph import run_test_orchestration


def create_app() -> FastAPI:
    app = FastAPI(title="Pet Insurance Recommender")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/test")
    def test() -> dict[str, str]:
        message = run_test_orchestration()
        return {"message": message}

    return app


app = create_app()
