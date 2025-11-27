import asyncio
import os

from crewai import Agent, Task, Crew, Process

from exam import DIR_ROOT
from exam.agents import createAgents
from exam.llm_provider import get_llm
from exam.tasks import create_assessment_tasks


class CrewAIExamClient:
    """
    Multi-Agent CrewAI client
    """
    def __init__(self):

        self.llm_config = get_llm()
        self.agents = createAgents(self.llm_config)
        self.evaluations_dir = DIR_ROOT / "evaluations"
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)

    async def full_exam(
            self,
            exam_date: str,
    ) -> dict:

        print("Multi-Agent Exam ASSESSMENT CrewAI")

        crew = Crew(
            agents=self.agents,
            tasks=create_assessment_tasks(self.agents,exam_date),
            llm=self.llm_config,
            verbose=False
        )

        result = await crew.kickoff_async()

        # EXTRACT TOKEN METRICS
        usage = result.token_usage
        print(f"\n[COST] Total Tokens: {usage.total_tokens}")
        print(f"[COST] Prompt Tokens: {usage.prompt_tokens}")
        print(f"[COST] Completion Tokens: {usage.completion_tokens}")

        return {
            "status": "success",
            "exam_date": exam_date,
            "result": result,
            "tokens": usage
        }

async def main():

    print(" CREWAI EXAM ASSESSMENT SYSTEM")
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