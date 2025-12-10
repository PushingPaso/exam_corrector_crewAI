import asyncio
import os
import time
import mlflow
from crewai import Crew

from exam import DIR_ROOT
from exam.agents import createAgents
from exam.llm_provider import get_llm
from exam.mlflow import calculate_overhead
from exam.tasks import create_assessment_tasks


class CrewAIExamClient:
    def __init__(self):
        self.llm_config = get_llm()
        self.agents = createAgents(self.llm_config)
        self.evaluations_dir = DIR_ROOT / "evaluations"
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)

    async def full_exam(self, exam_date: str) -> dict:

        mlflow.set_tracking_uri("http://localhost:5000")
        experiment = mlflow.set_experiment("CrewAI_Exam_Assessment")
        mlflow.crewai.autolog()

        with mlflow.start_run() as run:
            mlflow.log_param("framework", "CrewAI")
            mlflow.log_param("exam_date", exam_date)
            mlflow.log_param("model_name", getattr(self.llm_config, "model", "unknown"))

            print("Multi-Agent Exam ASSESSMENT CrewAI")

            crew = Crew(
                agents=self.agents,
                tasks=create_assessment_tasks(self.agents, exam_date),
                llm=self.llm_config,
                verbose=False
            )

            start_time = time.time()
            result = await crew.kickoff_async()
            end_time = time.time()
            duration = end_time - start_time

            usage = result.token_usage
            total_tokens = usage.total_tokens
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens


            print(f"Total Tokens: {total_tokens}")
            print(f"Prompt Tokens: {prompt_tokens}")
            print(f"Completion Tokens: {completion_tokens}")
            print(f"Duration (seconds): {duration:.2f}")

            mlflow.log_metric("total_tokens", total_tokens)
            mlflow.log_metric("prompt_tokens", prompt_tokens)
            mlflow.log_metric("completion_tokens", completion_tokens)
            mlflow.log_metric("duration_seconds", duration)

            calculate_overhead(run.info.run_id, duration)


            return {
                "status": "success",
                "exam_date": exam_date,
                "result": result,
                "tokens": usage
            }


async def main():
    print("CREWAI EXAM ASSESSMENT SYSTEM")
    print("\nThis system uses CrewAI multi-agent orchestration to:")
    print("  - Load exam data and checklists")
    print("  - Assess student answers with specialized agents")

    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

    client = CrewAIExamClient()
    exam_date = input("\nEnter exam date (format: YYYY-MM-DD, e.g., 2025-06-05): ").strip()
    await client.full_exam(exam_date)

    print("\nAll done!")


if __name__ == "__main__":
    asyncio.run(main())