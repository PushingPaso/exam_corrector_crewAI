from crewai import Agent
from exam.mcp import ExamMCPServer

mcp = ExamMCPServer()


def createAgents(llm_config: dict) -> list[Agent]:
    # --- CORREZIONE QUI ---
    # Assegna il riferimento alla funzione, non eseguirla.
    # L'agente la chiamerà con gli ID giusti in un secondo momento.
    load_checklist_tool = mcp.load_checklist
    # ----------------------

    load_exam_tool = mcp.load_exam_from_yaml_tool
    assess_feature_tool = mcp.assess_student_exam

    loader_agent = Agent(
        role="Exam Data Loader",
        goal="Load exam questions, student responses, and assessment checklists efficiently",
        backstory="""You are a meticulous data manager for the Software Engineering course.
            Your job is to load and organize exam data, ensuring all required information
            is available for the assessment team.""",
        tools=[load_checklist_tool, load_exam_tool],
        llm=llm_config,
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
        llm=llm_config,
        verbose=True,
        allow_delegation=False
    )

    # AGENT 3: Report Generator
    reporter_agent = Agent(
        role="Assessment Reporter",
        goal="Generate clear, comprehensive reports of assessment results",
        backstory="""You are a reporting specialist who creates detailed summaries
            of exam assessments. You organize results clearly and highlight key statistics.""",
        llm=llm_config,
        verbose=True,
        allow_delegation=False
    )

    return [loader_agent, assessor_agent, reporter_agent]


def getWorkerAgents(num_workers: int, llm_config: dict) -> list[Agent]:
    workers = []
    # (Questa funzione era già corretta)
    assess_feature_tool = mcp.assess_student_exam
    for i in range(num_workers):
        workers.append(Agent(
            role="Answer Assessor",
            goal="Evaluate student answers against assessment criteria with fairness and consistency",
            backstory="""You are an experienced Software Engineering professor.
            You evaluate student answers by checking if core concepts and important details
            are present. You are thorough but fair, giving credit where due.""",
            tools=[assess_feature_tool],
            llm=llm_config,
            verbose=True,
            allow_delegation=False))
    return workers