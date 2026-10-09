"""
核心Agent封装
基于LangChain Agent，集成记忆、结构化工具调用、反思功能
支持复杂任务的自动规划与多步骤执行
"""
import os
import uuid
from typing import Optional, List, Dict, Any
import logging

from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi

from config.settings import settings
from .memory_manager import MemoryManager
from .tool_registry import ToolRegistry
from .reflection import Reflection
from .planner import TaskPlanner
from llm_utils import message_to_text

logger = logging.getLogger(__name__)


class OfficeAgent:
    SYSTEM_PROMPT = """你是企业智能办公助手。

请根据用户问题选择合适的工具。工具执行结果是事实依据，不要编造工具未返回的信息。
如果问题不需要工具，直接回答；如果需要工具，先调用工具，再根据工具结果给出清晰的最终答案。"""

    def __init__(self, llm: Optional[ChatTongyi] = None, enable_planner: bool = True):
        if llm:
            self.llm = llm
        else:
            self._init_llm()

        self.memory_manager = MemoryManager()
        self.tool_registry = ToolRegistry()
        self.reflection = Reflection(self.llm)
        self.enable_planner = enable_planner and (self.llm is not None)
        if self.enable_planner:
            self.task_planner = TaskPlanner(self.llm)
        logger.info(f"OfficeAgent初始化完成，Planner模式: {'开启' if self.enable_planner else '关闭'}")

    def _init_llm(self):
        if settings.DASHSCOPE_API_KEY:
            os.environ["DASHSCOPE_API_KEY"] = settings.DASHSCOPE_API_KEY
            self.llm = ChatTongyi(
                model_name=settings.LLM_MODEL_NAME,
                model_kwargs={
                    "temperature": settings.LLM_TEMPERATURE,
                    "max_tokens": settings.LLM_MAX_TOKENS
                }
            )
            logger.info(f"LLM初始化成功，模型: {settings.LLM_MODEL_NAME}")
        else:
            logger.warning("未配置API Key，Agent将无法正常工作")
            self.llm = None

    def register_tool(self, tool):
        self.tool_registry.register(tool)

    def register_tools(self, tools: List):
        self.tool_registry.register_many(tools)

    def _create_agent(self):
        tools = self.tool_registry.list_tools()
        if not tools:
            raise ValueError("没有注册任何工具，请先注册工具")

        return create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self.SYSTEM_PROMPT,
            debug=settings.DEBUG,
            name="enterprise_office_agent"
        )

    @staticmethod
    def _extract_tool_steps(messages: List[Any]) -> List[Dict[str, Any]]:
        steps = []
        for index, message in enumerate(messages):
            for tool_call in getattr(message, "tool_calls", []) or []:
                observation = ""
                for candidate in messages[index + 1:]:
                    if getattr(candidate, "tool_call_id", None) == tool_call.get("id"):
                        observation = message_to_text(candidate)
                        break
                steps.append({
                    "tool": tool_call.get("name", "unknown"),
                    "tool_input": tool_call.get("args", {}),
                    "observation": observation
                })
        return steps

    def _is_complex_task(self, user_input: str) -> bool:
        complex_keywords = ["分析", "报告", "生成", "图表", "可视化", "工作总结", "会议纪要", "审批"]
        return any(kw in user_input for kw in complex_keywords)

    def _invoke_tool_with_reflection(self, tool, tool_name: str, params: dict):
        """工具失败后，调用 Reflection 判断是否重试。"""
        current_params = dict(params)
        for retry_count in range(self.reflection.max_retries + 1):
            try:
                return tool.invoke(current_params), current_params
            except Exception as e:
                error = str(e)
                adjusted_task = self.reflection.analyze_failure(
                    {"tool": tool_name, "params": current_params}, error
                )
                if not self.reflection.should_retry(error, retry_count):
                    raise
                if isinstance(adjusted_task, dict):
                    adjusted_params = adjusted_task.get("params")
                    if isinstance(adjusted_params, dict):
                        current_params = adjusted_params
                logger.info(
                    f"步骤工具 {tool_name} 反思后重试 ({retry_count + 1})"
                )

    def _execute_planned_task(self, user_input: str, session_id: str,
                               user_id: Optional[str] = None) -> Dict[str, Any]:
        """多步骤任务执行（使用规划器）"""
        available_tools = self.tool_registry.get_tool_names()
        task_steps = self.task_planner.plan(user_input, available_tools)
        if not task_steps:
            return self._execute_single_step(user_input, session_id, user_id)

        logger.info(f"任务规划成功，共 {len(task_steps)} 个步骤")
        all_steps = []
        final_output = ""

        for i, step in enumerate(task_steps):
            tool_name = step.get("tool")
            params = step.get("params", {})
            logger.info(f"执行步骤 {i+1}/{len(task_steps)}: {tool_name}")

            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                error_msg = f"错误：未找到工具 {tool_name}"
                all_steps.append({"step": i+1, "tool": tool_name, "error": error_msg})
                final_output += f"\n[步骤{i+1} 失败] {error_msg}"
                continue

            try:
                # 修复：使用标准调用 tool.invoke(params)
                observation, params = self._invoke_tool_with_reflection(
                    tool, tool_name, params
                )
                all_steps.append({
                    "tool": tool_name,
                    "tool_input": params,
                    "observation": observation
                })
                final_output += f"\n[步骤{i+1} 完成] {tool_name}: {observation[:200]}"
                self.memory_manager.mysql.log_tool_call(
                    session_id=session_id,
                    user_id=user_id or "anonymous",
                    tool_name=tool_name,
                    params=params,
                    result=str(observation)[:1000]
                )
            except Exception as e:
                error_msg = f"执行出错: {str(e)}"
                logger.error(f"步骤 {i+1} 执行失败: {e}")
                all_steps.append({"step": i+1, "tool": tool_name, "error": error_msg})
                final_output += f"\n[步骤{i+1} 出错] {error_msg}"

        reflection_summary = self.reflection.reflect_on_execution(
            task_steps,
            [
                step.get("observation", f"Error: {step.get('error', '')}")
                for step in all_steps
            ]
        )
        if reflection_summary:
            final_output += f"\n[反思总结] {reflection_summary}"

        return {
            "output": final_output.strip() or "任务执行完成",
            "intermediate_steps": all_steps,
            "session_id": session_id,
            "planned": True,
            "steps_count": len(task_steps),
            "reflection": reflection_summary
        }

    def _execute_single_step(self, user_input: str, session_id: str,
                             user_id: Optional[str] = None) -> Dict[str, Any]:
        """执行单步任务。"""
        try:
            agent = self._create_agent()
            user_context = ""
            if user_id:
                prefs = self.memory_manager.get_all_preferences(user_id)
                if prefs:
                    user_context = f"\n用户偏好信息：{prefs}"

            messages = [
                {"role": message["role"], "content": message["content"]}
                for message in self.memory_manager.get_messages(
                    session_id,
                    limit=settings.SHORT_TERM_MEMORY_MAX_LENGTH
                )
            ]
            messages.append({
                "role": "user",
                "content": user_input + user_context
            })

            result = agent.invoke({"messages": messages})
            result_messages = result.get("messages", [])
            steps = self._extract_tool_steps(result_messages)
            output = ""
            for message in reversed(result_messages):
                if getattr(message, "type", None) == "ai":
                    output = message_to_text(message)
                    if output:
                        break

            self.memory_manager.add_message(session_id, "user", user_input)
            if output:
                self.memory_manager.add_message(session_id, "assistant", output)

            for step in steps:
                self.memory_manager.mysql.log_tool_call(
                    session_id=session_id,
                    user_id=user_id or "anonymous",
                    tool_name=step["tool"],
                    params=step["tool_input"],
                    result=str(step["observation"])[:1000]
                )
            return {
                "output": output or "执行完成",
                "intermediate_steps": steps,
                "session_id": session_id,
                "planned": False
            }
        except Exception as e:
            logger.error(f"单步执行失败: {e}", exc_info=True)
            return {
                "output": f"执行失败: {str(e)}",
                "intermediate_steps": [],
                "session_id": session_id,
                "error": str(e)
            }

    def run(self, user_input: str, session_id: Optional[str] = None,
            user_id: Optional[str] = None, use_planner: bool = None) -> Dict[str, Any]:
        if session_id is None:
            session_id = str(uuid.uuid4())
        if not self.llm:
            return {
                "output": "LLM未初始化，请配置API Key后重试。",
                "intermediate_steps": [],
                "session_id": session_id
            }
        if use_planner is None:
            use_planner = self.enable_planner and self._is_complex_task(user_input)
        if use_planner:
            logger.info("使用任务规划器处理请求")
            return self._execute_planned_task(user_input, session_id, user_id)
        else:
            logger.info("使用tool-calling Agent处理请求")
            return self._execute_single_step(user_input, session_id, user_id)

    def clear_session(self, session_id: str):
        self.memory_manager.clear_session(session_id)
