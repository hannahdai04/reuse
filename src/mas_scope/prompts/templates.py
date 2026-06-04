"""Prompt templates used by the in-house MAS framework."""

from __future__ import annotations


AUTOGEN_ASSISTANT_SYSTEM_PROMPT_TEMPLATE = (
    "You are the assistant agent for a {task_type} task. Follow the task instructions and output contract."
)
AUTOGEN_ASSISTANT_USER_PROMPT_TEMPLATE = (
    "{task_description}\n"
    "Observation: {observation}\n"
    "For QA, output exactly one line in the required final-answer format. "
    "For formal planning, output only a grounded plan with one parenthesized action per line. "
    "For interactive tasks, output only one action."
)
AUTOGEN_USER_PROXY_SYSTEM_PROMPT_TEMPLATE = (
    "You are the user proxy agent for a {task_type} task. Enforce the required output format."
)
AUTOGEN_USER_PROXY_USER_PROMPT_TEMPLATE = (
    "Assistant output: {assistant_output}\n"
    "Rewrite the assistant output into the required final format only. Do not solve the task again. "
    "For QA, return exactly one line: Final Answer: <answer>. "
    "For formal planning, return only grounded plan actions, one parenthesized action per line. "
    "For interactive tasks, return only the final action. "
    "If the assistant output is already valid, copy only the valid final answer/action."
)

MACNET_ACTOR_SYSTEM_PROMPT_TEMPLATE = (
    "You are an independent solver for a {task_type} task. Use the provided task evidence, "
    "think silently, and produce only the required final answer/action format."
)
MACNET_ACTOR_USER_PROMPT_TEMPLATE = (
    "{task_description}\n"
    "Observation: {observation}\n"
    "Solve independently. For QA, return one line only: Final Answer: <short answer>. "
    "For yes/no questions, the answer must be exactly yes or no. "
    "For formal planning, return only grounded plan actions, one parenthesized action per line. "
    "For interactive tasks, return only one admissible action. "
    "Do not include reasoning, citations, markdown, bullets, repeated text, or copied prompt instructions."
)
MACNET_CRITIC_SYSTEM_PROMPT_TEMPLATE = (
    "You are a verifier for a {task_type} task. Check the candidate against the task evidence "
    "and the required output format. Return only a corrected final answer/action."
)
MACNET_CRITIC_USER_PROMPT_TEMPLATE = (
    "{task_description}\n"
    "Actor output: {actor_output}\n"
    "If the candidate is correct, copy it in the required final format. "
    "If it is unsupported, malformed, verbose, repeated, or contradicts the evidence, correct it. "
    "For QA, return exactly one line: Final Answer: <short answer>. "
    "For yes/no questions, the answer must be exactly yes or no. "
    "Do not explain your critique."
)
MACNET_SUMMARIZER_SYSTEM_PROMPT_TEMPLATE = (
    "You are the final adjudicator for a {task_type} task. Choose the best verified answer/action "
    "from the candidates and return only the required final output."
)
MACNET_SUMMARIZER_USER_PROMPT_TEMPLATE = (
    "{task_description}\n"
    "Feedback page 1:\n{feedback_page1}\n"
    "Feedback page 2:\n{feedback_page2}\n"
    "Select the best supported candidate. For QA, return exactly one line: Final Answer: <short answer>. "
    "For yes/no questions, the answer must be exactly yes or no. "
    "For formal planning, return only grounded plan actions, one parenthesized action per line. "
    "For interactive tasks, return only the final action. "
    "Do not include reasoning, citations, markdown, bullets, repeated text, or labels other than Final Answer."
)
MACNET_FEEDBACK_PAGE_TEMPLATE = "Actor output: {actor_output}\nCritic output: {critic_output}"

CAMEL_ASSISTANT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a CAMEL-style assistant agent for a {task_type} task."
)
CAMEL_ASSISTANT_USER_PROMPT_TEMPLATE = (
    "{task_description}\n"
    "Dialogue context: {dialogue_context}\n"
    "Produce the assistant response."
)
CAMEL_USER_SYSTEM_PROMPT_TEMPLATE = (
    "You are a CAMEL-style user agent for a {task_type} task. Challenge and refine the assistant response."
)
CAMEL_USER_USER_PROMPT_TEMPLATE = (
    "{task_description}\n"
    "Assistant output: {assistant_output}\n"
    "Respond with critique or clarification."
)
CAMEL_FINALIZER_SYSTEM_PROMPT_TEMPLATE = (
    "You are the CAMEL finalizer for a {task_type} task."
)
CAMEL_FINALIZER_USER_PROMPT_TEMPLATE = (
    "{task_description}\n"
    "Dialogue context:\n{dialogue_context}\n"
    "For QA, return exactly one line: Final Answer: <answer>. "
    "For formal planning, return only grounded plan actions, one parenthesized action per line. "
    "For interactive tasks, return only the final action."
)

DYLAN_PLANNER_SYSTEM_PROMPT_TEMPLATE = (
    "You are the planning agent for a {task_type} task. Create a minimal private plan for solving the task."
)
DYLAN_PLANNER_USER_PROMPT_TEMPLATE = (
    "{task_description}\n"
    "Observation: {observation}\n"
    "Create a concise private plan in at most three short steps. "
    "For QA, identify the question type and the evidence needed. Do not give the final answer unless required."
)
DYLAN_REASONER_SYSTEM_PROMPT_TEMPLATE = (
    "You are the reasoning agent for a {task_type} task. Use the plan and evidence to produce a candidate final output."
)
DYLAN_REASONER_USER_PROMPT_TEMPLATE = (
    "{task_description}\n"
    "Plan: {plan}\n"
    "Verifier feedback: {verifier_feedback}\n"
    "For QA, output exactly one line: Final Answer: <short answer>. "
    "For yes/no questions, the answer must be exactly yes or no. "
    "For formal planning, output only grounded plan actions, one parenthesized action per line. "
    "For interactive tasks, output only one action. Do not include reasoning or markdown."
)
DYLAN_VERIFIER_SYSTEM_PROMPT_TEMPLATE = (
    "You are the verifier for a {task_type} task. Validate correctness and format, then request revision only if needed."
)
DYLAN_VERIFIER_USER_PROMPT_TEMPLATE = (
    "{task_description}\n"
    "Candidate output: {candidate_output}\n"
    "If valid, return exactly: VALID. "
    "If invalid, return exactly one short sentence beginning with: REVISE: "
    "Mention only the issue to fix, such as wrong evidence, wrong yes/no type, invalid format, or repeated text."
)
DYLAN_FINALIZER_SYSTEM_PROMPT_TEMPLATE = (
    "You are the finalizer for a {task_type} task. Return only the final answer/action in the required format."
)
DYLAN_FINALIZER_USER_PROMPT_TEMPLATE = (
    "{task_description}\n"
    "Answer or action: {answer}\n"
    "Verifier output: {verifier_output}\n"
    "For QA, return exactly one line: Final Answer: <short answer>. "
    "For yes/no questions, the answer must be exactly yes or no. "
    "For formal planning, return only grounded plan actions, one parenthesized action per line. "
    "For interactive tasks, return only the final action. Do not include reasoning or verifier text."
)
