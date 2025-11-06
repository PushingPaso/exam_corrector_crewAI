"""
Sistema di valutazione esami con CrewAI.
Converte il sistema LangChain in un team di agenti CrewAI.
"""

import json
from typing import List

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from pydantic import BaseModel, Field

# Import esistenti
from exam import DIR_ROOT, get_questions_store
from exam.solution import load_cache as load_answer_cache

from exam.llm_provider import get_llm, get_llm_config
# ============================================================================


# ============================================================================
# MODELLI PYDANTIC (invariati)
# ============================================================================

class FeatureAssessment(BaseModel):
    satisfied: bool = Field(description="Whether the feature is present in the answer")
    motivation: str = Field(description="Explanation of why the feature is present or not")


# ============================================================================
# CREWAI TOOLS (ex LangChain tools)
# ============================================================================

@tool("Load Checklist")
def load_checklist_tool(question_id: str) -> str:
    """
    Load the assessment checklist for a question.

    Args:
        question_id: The question ID (e.g., "CI-5")

    Returns:
        JSON string with checklist details
    """
    try:
        questions_store = get_questions_store()
        question = questions_store.question(question_id)
        checklist = load_answer_cache(question)

        if not checklist:
            return json.dumps({"error": f"No checklist found for {question_id}"})

        return json.dumps({
            "status": "success",
            "question_id": question_id,
            "question_text": question.text,
            "core_items": checklist.core,
            "important_items": checklist.details_important,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Load Exam from YAML")
def load_exam_tool(questions_file: str, responses_file: str, grades_file: str = None) -> str:
    """
    Load an entire exam from YAML files.

    Args:
        questions_file: Questions YAML filename
        responses_file: Responses YAML filename
        grades_file: Optional grades YAML filename

    Returns:
        JSON string with exam data
    """
    try:
        from exam import load_exam_from_yaml

        exam_data = load_exam_from_yaml(
            questions_file=questions_file,
            responses_file=responses_file,
            grades_file=grades_file
        )

        return json.dumps({
            "status": "success",
            "exam_id": exam_data["exam_id"],
            "num_questions": len(exam_data["questions"]),
            "num_students": len(exam_data["students"]),
            "questions": exam_data["questions"],
            "students": [
                {
                    "email": s["email"],
                    "num_responses": s["num_responses"]
                }
                for s in exam_data["students"]
            ]
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Assess Single Feature")
def assess_feature_tool(
        question_text: str,
        feature_description: str,
        feature_type: str,
        student_response: str
) -> str:
    """
    Assess a single feature in a student's answer.

    Args:
        question_text: The question text
        feature_description: What to check for
        feature_type: "core" or "important detail"
        student_response: Student's answer

    Returns:
        JSON with assessment result
    """

    try:
        llm = get_llm()

        # Crea il prompt
        prompt = f"""You are a teacher evaluating a student's answer.

Question: {question_text}

Feature to check ({feature_type}): {feature_description}

Student's answer: {student_response}

Determine if this feature is present and explain why.
Return JSON with 'satisfied' (bool) and 'motivation' (string)."""

        # Invoca il modello con structured output
        from langchain_core.output_parsers import PydanticOutputParser
        parser = PydanticOutputParser(pydantic_object=FeatureAssessment)

        prompt_with_format = prompt + f"\n\n{parser.get_format_instructions()}"
        result = llm.invoke(prompt_with_format)

        # Parse il risultato
        assessment = parser.parse(result.content)

        return json.dumps({
            "satisfied": assessment.satisfied,
            "motivation": assessment.motivation
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# CREWAI AGENTS (ex AIOracle/Assessor)
# ============================================================================

def create_assessment_agents(llm_config: dict) -> tuple[Agent, Agent, Agent]:
    """
    Crea gli agenti specializzati per la valutazione.

    Returns:
        Tuple di (loader_agent, assessor_agent, reporter_agent)
    """

    # AGENT 1: Data Loader
    loader_agent = Agent(
        role="Exam Data Loader",
        goal="Load exam questions, student responses, and assessment checklists efficiently",
        backstory="""You are a meticulous data manager for the Software Engineering course.
        Your job is to load and organize exam data, ensuring all required information
        is available for the assessment team.""",
        tools=[load_checklist_tool, load_exam_tool],
        llm=llm_config["llm"],
        verbose=True,
        allow_delegation=False
    )

    # AGENT 2: Answer Assessor
    assessor_agent = Agent(
        role="Answer Assessor",
        goal="Evaluate student answers against assessment criteria with fairness and consistency",
        backstory="""You are an experienced Software Engineering professor.
        You evaluate student answers by checking if core concepts and important details
        are present. You are thorough but fair, giving credit where due.""",
        tools=[assess_feature_tool],
        llm=llm_config["llm"],
        verbose=True,
        allow_delegation=False
    )

    # AGENT 3: Report Generator
    reporter_agent = Agent(
        role="Assessment Reporter",
        goal="Generate clear, comprehensive reports of assessment results",
        backstory="""You are a reporting specialist who creates detailed summaries
        of exam assessments. You organize results clearly and highlight key statistics.""",
        llm=llm_config["llm"],
        verbose=True,
        allow_delegation=False
    )

    return loader_agent, assessor_agent, reporter_agent


# ============================================================================
# CREWAI TASKS (ex workflow steps)
# ============================================================================

def create_assessment_tasks(
        agents: tuple[Agent, Agent, Agent],
        exam_date: str,
        student_email: str = None
) -> List[Task]:
    """
    Crea le task per il processo di valutazione.

    Args:
        agents: Tuple di (loader, assessor, reporter)
        exam_date: Data dell'esame (es. "2025-06-05")
        student_email: Email studente (None = tutti)

    Returns:
        Lista di Task CrewAI
    """
    loader_agent, assessor_agent, reporter_agent = agents

    # TASK 1: Load Exam
    load_task = Task(
        description=f"""Load the exam from date {exam_date}.

        Steps:
        1. Load questions from: se-{exam_date}-questions.yml
        2. Load responses from: se-{exam_date}-responses.yml
        3. Load grades from: se-{exam_date}-grades.yml (if available)
        4. Load checklists for each question

        Return a summary with number of questions and students loaded.""",
        expected_output="JSON summary of loaded exam data",
        agent=loader_agent
    )

    # TASK 2: Assess Answers
    if student_email:
        assess_desc = f"Assess all answers for student: {student_email}"
    else:
        assess_desc = "Assess all answers for all students in the exam"

    assess_task = Task(
        description=f"""{assess_desc}

        For each student response:
        1. Check each core element (70% weight)
        2. Check each important detail (30% weight)
        3. Calculate the final score
        4. Provide constructive feedback

        Use the assess_feature_tool for each feature check.
        Be systematic and thorough.""",
        expected_output="Complete assessment results for all evaluated answers",
        agent=assessor_agent,
        context=[load_task]  # Depends on load_task
    )

    # TASK 3: Generate Report
    report_task = Task(
        description="""Generate a comprehensive assessment report.

        Include:
        1. Overall statistics (average, min, max scores)
        2. Individual student results
        3. Comparison with original Moodle grades (if available)
        4. Score breakdown by question

        Format the report clearly and professionally.""",
        expected_output="Markdown formatted assessment report",
        agent=reporter_agent,
        context=[assess_task]  # Depends on assess_task
    )

    return [load_task, assess_task, report_task]


# ============================================================================
# CREWAI CREW (ex AgentExecutor/LangGraph)
# ============================================================================

class ExamAssessmentCrew:
    """
    Crew principale per la valutazione esami.
    Sostituisce AgentExecutor e LangGraph orchestration.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):

        self.llm_config = get_llm_config(model_name)
        self.agents = create_assessment_agents(self.llm_config)
        self.evaluations_dir = DIR_ROOT / "evaluations"
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)

    def assess_exam(
            self,
            exam_date: str,
            student_email: str = None,
            process_type: Process = Process.sequential
    ) -> dict:
        """
        Valuta un esame completo.

        Args:
            exam_date: Data esame (YYYY-MM-DD)
            student_email: Email studente specifico (None = tutti)
            process_type: Process.sequential o Process.hierarchical

        Returns:
            Risultati della valutazione
        """
        # Crea le task
        tasks = create_assessment_tasks(self.agents, exam_date, student_email)

        # Crea il Crew
        crew = Crew(
            agents=list(self.agents),
            tasks=tasks,
            process=process_type,
            verbose=True
        )

        # Esegui il workflow
        print(f"\n{'=' * 70}")
        print(f"CREWAI ASSESSMENT: Exam {exam_date}")
        if student_email:
            print(f"Student: {student_email[:30]}...")
        print(f"{'=' * 70}\n")

        result = crew.kickoff()

        print(f"\n{'=' * 70}")
        print("ASSESSMENT COMPLETED")
        print(f"{'=' * 70}\n")

        return {
            "status": "success",
            "exam_date": exam_date,
            "result": result
        }

    def assess_exam_parallel(
            self,
            exam_date: str,
            num_workers: int = 3
    ) -> dict:
        """
        Valuta esame con workers paralleli.
        In CrewAI, usiamo Process.hierarchical per parallelismo.

        Args:
            exam_date: Data esame
            num_workers: Numero di worker paralleli

        Returns:
            Risultati aggregati
        """
        print(f"\n{'=' * 70}")
        print(f"PARALLEL ASSESSMENT with {num_workers} workers")
        print(f"{'=' * 70}\n")

        # In CrewAI, il parallelismo è gestito con Process.hierarchical
        # e un manager agent che delega ai worker

        # Crea manager agent
        manager = Agent(
            role="Assessment Manager",
            goal="Coordinate parallel assessment of multiple students",
            backstory="You manage a team of assessors to evaluate exams efficiently",
            llm=self.llm_config["llm"],
            verbose=True
        )

        # Crea worker agents
        workers = [
            Agent(
                role=f"Assessment Worker {i + 1}",
                goal="Assess assigned student responses quickly and accurately",
                backstory=f"You are worker #{i + 1} in the assessment team",
                tools=[assess_feature_tool],
                llm=self.llm_config["llm"],
                verbose=True
            )
            for i in range(num_workers)
        ]

        # Crea task principale
        main_task = Task(
            description=f"""Assess all students in exam {exam_date}.

            Delegate work evenly among your {num_workers} workers.
            Each worker should assess approximately the same number of students.
            Collect and aggregate results.""",
            expected_output="Aggregated assessment results for all students",
            agent=manager
        )

        # Crea crew gerarchico
        crew = Crew(
            agents=[manager] + workers,
            tasks=[main_task],
            process=Process.hierarchical,
            manager_llm=self.llm_config["llm"],
            verbose=True # Corretto da '2' a 'True'
        )

        result = crew.kickoff()

        return {
            "status": "success",
            "exam_date": exam_date,
            "num_workers": num_workers,
            "result": result
        }


# ============================================================================
# FUNZIONI DI COMPATIBILITÀ
# ============================================================================

async def assess_student_exam(
        student_email: str,
        exam_questions: list,
        student_responses: dict,
        questions_store,
        context,
        save_results: bool = True,
        original_grades: dict = None
) -> dict:
    """
    Funzione di compatibilità con il codice esistente.
    Usa CrewAI internamente.
    """
    # Estrai exam_date dal context o dai dati
    exam_date = "2025-06-05"  # Default, dovrebbe essere passato

    crew = ExamAssessmentCrew()
    result = crew.assess_exam(exam_date, student_email)

    # Formatta risultato in formato compatibile
    return {
        "student_email": student_email,
        "calculated_score": 0.0,  # TODO: parse from result
        "max_score": sum(q["score"] for q in exam_questions),
        "percentage": 0.0,
        "scoring_system": "70% Core + 30% Important_Details",
        "assessments": [],
        "original_grades": original_grades if original_grades else {}
    }