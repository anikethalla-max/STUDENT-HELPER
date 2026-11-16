"""Shared data passed between the learning workflow's nodes."""

from operator import add
from typing import Annotated, TypedDict

from agents.assessment import AssessmentFeedback, PracticeQuestion
from agents.planner import PlannerDecision
from agents.research import ResearchResult
from agents.tutor import Lesson
from learn_topic.chat_memory import ChatMessage
from learn_topic.retrieval import StudyDocument


class WorkflowState(TypedDict, total=False):
    request: str
    level: str
    notes: str
    documents: list[StudyDocument]
    notes_only: bool
    thread_context: str
    chat_messages: Annotated[list[ChatMessage], add]
    learning_progress: str
    material_summaries: tuple[str, ...]
    closed: bool
    decision: PlannerDecision
    approved: bool | None
    lesson: Lesson | None
    research: ResearchResult | None
    question: PracticeQuestion | None
    answer: str | None
    feedback: AssessmentFeedback | None
