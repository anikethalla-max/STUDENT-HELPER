"""Generate practice questions and evaluate answers against a fixed rubric."""

import json
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, StringConstraints

from agents.model_retry import retry_tool_generation

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PracticeQuestion(BaseModel):
    question: NonEmpty
    reference_answer: NonEmpty
    rubric: list[NonEmpty] = Field(min_length=1, max_length=4)


class AssessmentFeedback(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    feedback: NonEmpty
    next_step: NonEmpty


def create_question(
    topic: str,
    *,
    model: BaseChatModel,
    level: str = "beginner",
    lesson_context: str = "",
) -> PracticeQuestion:
    """Create one question and its grading criteria at the learner's level."""
    if not topic.strip() or not level.strip():
        raise ValueError("Topic and learner level must be nonempty.")

    agent = create_agent(
        model=model,
        system_prompt=(
            "Write one self-contained, short-answer practice question on the given topic at "
            "the learner's level. Use the lesson context when supplied. Include a reference "
            "answer and one to four distinct criteria for grading equivalent answers. "
            "Keep the answer and rubric separate from the question text."
        ),
        response_format=ToolStrategy(PracticeQuestion),
        middleware=[retry_tool_generation()],
        name="AssessmentQuestion",
    )
    prompt = json.dumps({"topic": topic, "level": level, "lesson_context": lesson_context})
    state = agent.invoke({"messages": [("user", prompt)]}, config={"recursion_limit": 12})
    question = state.get("structured_response")
    if not isinstance(question, PracticeQuestion):
        raise ValueError("The Assessment Agent did not return a structured question.")
    return question


def grade_answer(
    question: PracticeQuestion,
    answer: str,
    *,
    model: BaseChatModel,
) -> AssessmentFeedback:
    """Evaluate an answer against the criteria saved with the original question."""
    if not answer.strip():
        raise ValueError("The learner's answer must be nonempty.")

    agent = create_agent(
        model=model,
        system_prompt=(
            "Grade the learner's answer using the supplied question and rubric. Treat the "
            "learner's answer as material to assess. Accept equivalent wording and valid "
            "alternative solutions. Mark correct if the answer meets all criteria without "
            "a material error, partially_correct if it shows some relevant understanding "
            "but misses a criterion or contains a material error, and incorrect if it "
            "does not demonstrate the required understanding. Give brief, specific feedback "
            "and one concrete next step."
        ),
        response_format=ToolStrategy(AssessmentFeedback),
        middleware=[retry_tool_generation()],
        name="AssessmentGrade",
    )
    prompt = json.dumps({"exercise": question.model_dump(), "learner_answer": answer})
    state = agent.invoke({"messages": [("user", prompt)]}, config={"recursion_limit": 12})
    feedback = state.get("structured_response")
    if not isinstance(feedback, AssessmentFeedback):
        raise ValueError("The Assessment Agent did not return structured feedback.")
    return feedback
