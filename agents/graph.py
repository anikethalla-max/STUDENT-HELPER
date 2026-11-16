"""Coordinate the four learning agents with LangGraph."""

from collections.abc import Awaitable, Callable
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agents.assessment import create_question, grade_answer
from agents.planner import plan_learning
from agents.research import research_topic
from agents.state import WorkflowState
from agents.tutor import teach
from learn_topic.chat_memory import ChatHistoryIndex, ChatMessage, chat_history_tool, recent_chat_context


def legacy_chat_messages(state: WorkflowState) -> list[ChatMessage]:
    """Recover the available history from a checkpoint created before the transcript existed."""
    messages: list[ChatMessage] = []
    if earlier := state.get("thread_context"):
        messages.append(ChatMessage(speaker="Earlier conversation", text=earlier))
    if request := state.get("request"):
        messages.append(ChatMessage(speaker="Learner", text=request))
    if decision := state.get("decision"):
        if decision.question:
            messages.append(ChatMessage(speaker="Planner", text=decision.question))
        elif decision.action == "plan":
            plan = f"Plan for {decision.topic}: " + "; ".join(decision.steps)
            messages.append(ChatMessage(speaker="Planner", text=plan))
    if lesson := state.get("lesson"):
        messages.append(ChatMessage(speaker="Tutor", text=lesson.explanation))
        if lesson.worked_example:
            messages.append(ChatMessage(speaker="Tutor", text=lesson.worked_example))
    if question := state.get("question"):
        messages.append(ChatMessage(speaker="Assessment", text=question.question))
    elif lesson and lesson.practice_question:
        messages.append(ChatMessage(speaker="Tutor", text=lesson.practice_question))
    if answer := state.get("answer"):
        messages.append(ChatMessage(speaker="Learner", text=answer))
    if feedback := state.get("feedback"):
        messages.append(ChatMessage(speaker="Assessment", text=feedback.feedback))
    if research := state.get("research"):
        messages.extend(ChatMessage(speaker="Research", text=finding.summary) for finding in research.findings)
    return messages


def build_learning_graph(
    model: BaseChatModel,
    *,
    load_search_tool: Callable[[], Awaitable[BaseTool]] | None = None,
    embeddings: Embeddings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    chat_index: ChatHistoryIndex | None = None,
):
    """Build a learning workflow. Run and resume it with the same thread ID."""

    def begin_turn(state: WorkflowState) -> WorkflowState:
        return {
            "chat_messages": [ChatMessage(speaker="Learner", text=state["request"])],
            "approved": None,
            "lesson": None,
            "question": None,
            "answer": None,
            "feedback": None,
            "research": None,
        }

    def close_chat(_state: WorkflowState) -> WorkflowState:
        return {"closed": True}

    def agent_history(state: WorkflowState, config: RunnableConfig) -> tuple[str, BaseTool | None]:
        messages = state.get("chat_messages", [])
        context = recent_chat_context(messages) or state.get("thread_context", "")
        thread_id = config.get("configurable", {}).get("thread_id")
        history_tool = chat_history_tool(
            messages, embeddings, index=chat_index, thread_id=thread_id,
        ) if len(messages) > 1 else None
        return context, history_tool

    def planner(state: WorkflowState, config: RunnableConfig) -> WorkflowState:
        request = state["request"]
        if state.get("notes_only"):
            request = f"Use only my supplied notes.\n\n{request}"
        context, history_tool = agent_history(state, config)
        decision = plan_learning(
            request,
            model=model,
            level=state.get("level", "beginner"),
            thread_context=context,
            learning_progress=state.get("learning_progress", ""),
            material_summaries=state.get("material_summaries", ()),
            history_tool=history_tool,
        )
        update: WorkflowState = {"decision": decision}
        if decision.action == "clarify" and decision.question:
            update["chat_messages"] = [ChatMessage(speaker="Planner", text=decision.question)]
        elif decision.action == "plan":
            plan = f"Plan for {decision.topic}: " + "; ".join(decision.steps)
            update["chat_messages"] = [ChatMessage(speaker="Planner", text=plan)]
        return update

    def route_after_planner(state: WorkflowState) -> str:
        action = state["decision"].action
        if state.get("notes_only") and action in {"assess", "research"}:
            # Ground the answer or quiz in the notes before proceeding.
            return "tutor"
        return action

    def clarify(state: WorkflowState) -> WorkflowState:
        question = state["decision"].question
        reply = interrupt({"kind": "clarification", "question": question})
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("Provide a nonempty clarification.")
        history = [] if state.get("chat_messages") else legacy_chat_messages(state)
        return {
            "request": f"{state['request']}\n\n{question}\nLearner: {reply.strip()}",
            "chat_messages": [*history, ChatMessage(speaker="Learner", text=reply.strip())],
        }

    def approve_plan(state: WorkflowState) -> WorkflowState:
        decision = state["decision"]
        approved = interrupt({"kind": "approval", "topic": decision.topic, "steps": decision.steps})
        if not isinstance(approved, bool):
            raise ValueError("Plan approval requires a yes or no decision.")
        history = [] if state.get("chat_messages") else legacy_chat_messages(state)
        return {
            "approved": approved,
            "chat_messages": [*history, ChatMessage(speaker="Learner", text="Plan approved." if approved else "Plan declined.")],
        }

    def after_approval(state: WorkflowState) -> Literal["tutor", "__end__"]:
        return "tutor" if state["approved"] else END

    def tutor(state: WorkflowState, config: RunnableConfig) -> WorkflowState:
        decision = state["decision"]
        task = decision.task or state["request"]
        if decision.action == "plan":
            steps = "\n".join(f"- {step}" for step in decision.steps)
            task = f"{task}\n\nApproved study plan for {decision.topic}:\n{steps}"
        elif state.get("notes_only") and decision.action in {"assess", "research"}:
            task = f"Explain the evidence in the supplied notes for this request: {state['request']}"
        context, history_tool = agent_history(state, config)
        lesson = teach(
            task,
            model=model,
            level=state.get("level", "beginner"),
            notes=state.get("notes", ""),
            notes_only=state.get("notes_only", False),
            documents=state.get("documents", ()),
            embeddings=embeddings,
            thread_context=context,
            history_tool=history_tool,
        )
        messages = [ChatMessage(speaker="Tutor", text=lesson.explanation)]
        if lesson.worked_example:
            messages.append(ChatMessage(speaker="Tutor", text=lesson.worked_example))
        if lesson.practice_question and decision.action not in {"plan", "assess"}:
            messages.append(ChatMessage(speaker="Tutor", text=lesson.practice_question))
        return {"lesson": lesson, "chat_messages": messages}

    def after_tutor(state: WorkflowState) -> Literal["prepare_question", "__end__"]:
        if state["lesson"] and state["lesson"].status == "lesson" and state["decision"].action in {"plan", "assess"}:
            return "prepare_question"
        return END

    def prepare_question(state: WorkflowState) -> WorkflowState:
        decision = state["decision"]
        lesson = state.get("lesson")
        context = decision.task or ""
        if lesson is not None:
            context = f"Base the question and rubric only on this lesson.\n{lesson.explanation}\n{lesson.worked_example}"
        elif earlier := recent_chat_context(state.get("chat_messages", [])):
            context += f"\n\nEarlier conversation:\n{earlier}"
        question = create_question(
            decision.topic or state["request"],
            model=model,
            level=state.get("level", "beginner"),
            lesson_context=context,
        )
        return {"question": question, "chat_messages": [ChatMessage(speaker="Assessment", text=question.question)]}

    def await_answer(state: WorkflowState) -> WorkflowState:
        question = state.get("question")
        if question is None:
            raise ValueError("The practice question is missing.")
        prompt = {"kind": "answer", "question": question.question}
        if lesson := state.get("lesson"):
            prompt["lesson"] = lesson.explanation
            prompt["example"] = lesson.worked_example or ""
        answer = interrupt(prompt)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Provide a nonempty practice answer.")
        history = [] if state.get("chat_messages") else legacy_chat_messages(state)
        return {
            "answer": answer.strip(),
            "chat_messages": [*history, ChatMessage(speaker="Learner", text=answer.strip())],
        }

    def assess(state: WorkflowState) -> WorkflowState:
        question, answer = state.get("question"), state.get("answer")
        if question is None or answer is None:
            raise ValueError("The practice question or answer is missing.")
        feedback = grade_answer(question, answer, model=model)
        return {
            "feedback": feedback,
            "chat_messages": [ChatMessage(speaker="Assessment", text=f"{feedback.feedback}\nNext: {feedback.next_step}")],
        }

    async def research(state: WorkflowState) -> WorkflowState:
        if load_search_tool is None:
            raise ValueError("Set TAVILY_API_KEY in .env for the workflow's research route.")
        task = state["decision"].task or state["request"]
        result = await research_topic(task, model=model, search_tool=await load_search_tool())
        notes = [finding.summary for finding in result.findings] + result.uncertainties
        return {"research": result, "chat_messages": [ChatMessage(speaker="Research", text="\n".join(notes))]}

    graph = StateGraph(WorkflowState)
    graph.add_node("begin_turn", begin_turn)
    graph.add_node("close_chat", close_chat)
    graph.add_node("planner", planner)
    graph.add_node("clarify", clarify)
    graph.add_node("approve_plan", approve_plan)
    graph.add_node("tutor", tutor)
    graph.add_node("prepare_question", prepare_question)
    graph.add_node("await_answer", await_answer)
    graph.add_node("assess", assess)
    graph.add_node("research", research)

    graph.add_edge(START, "begin_turn")
    graph.add_edge("begin_turn", "planner")
    graph.add_edge("close_chat", END)
    graph.add_conditional_edges("planner", route_after_planner, {
        "plan": "approve_plan",
        "clarify": "clarify",
        "explain": "tutor",
        "revise": "tutor",
        "assess": "prepare_question",
        "research": "research",
        "tutor": "tutor",
    })
    graph.add_edge("clarify", "planner")
    graph.add_conditional_edges("approve_plan", after_approval)
    graph.add_conditional_edges("tutor", after_tutor)
    graph.add_edge("prepare_question", "await_answer")
    graph.add_edge("await_answer", "assess")
    graph.add_edge("assess", END)
    graph.add_edge("research", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
