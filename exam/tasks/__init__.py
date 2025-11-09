
from crewai import Agent, Task, Crew, Process


def create_assessment_tasks(
        agents: list[Agent],
        exam_date: str,
        student_email: str = None
) -> list[Task]:
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
        1. Load questions from: se-{exam_date}-questions.yml, responses from: se-{exam_date}-responses.yml, load grades from: se-{exam_date}-grades.yml (if available)
        2. Load checklists for each question you found in question.yml passing the id to the relative tool""",
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
        Assess the exam answer of the student 
        Use the assess_feature_tool for each feature check.
        Be systematic and thorough.""",
        expected_output="Complete assessment results for all evaluated answers",
        agent=assessor_agent,
        context=[load_task]
    )

    return [load_task, assess_task]