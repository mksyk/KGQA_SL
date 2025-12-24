"""
Streamlit 版本的智能问答应用
基于 Flask 版本转换而来，保持后端逻辑不变
"""
import json
import os
from typing import Optional, Tuple, Dict
import requests
import streamlit as st
from user_storage import credentials, write_credentials, storage_file, Credentials

# 导入 NER 和 KG 检索模块
from ner_module import get_ner_model
from kg_retriever import get_kg_retriever
import config as app_config


# Dify API 配置 - 可以配置多个工作流
WORKFLOWS = {
    # "testflow": {
    #     "url": os.environ.get("DIFY_API_URL", "http://172.25.219.127/v1/chat-messages"),
    #     "key": os.environ.get("DIFY_API_KEY", "app-LUFceHJWRbUuhGBDD96VIsJX"),
    # },
    "agentflow1": {
        "url": os.environ.get("DIFY_API_URL", "http://172.25.219.127/v1/chat-messages"),
        "key": os.environ.get("DIFY_API_KEY", "app-is4ADPVdQhq5ArvSlm5CJty3"),
    }
}

USER_ID = os.environ.get("DIFY_USER_ID", "cuirj")


def get_default_workflow_name() -> str:
    """获取默认工作流名称（WORKFLOWS 中的第一个）"""
    if not WORKFLOWS:
        raise ValueError("未配置任何工作流")
    return next(iter(WORKFLOWS))


def format_kg_content(kg_results: Dict) -> str:
    """
    将知识图谱检索结果格式化为字符串
    
    Args:
        kg_results: 知识图谱检索结果字典
        
    Returns:
        格式化后的知识图谱内容字符串
    """
    if not kg_results or not kg_results.get("entities"):
        return "未找到相关知识图谱信息。"
    
    content_parts = []
    
    for entity_result in kg_results["entities"]:
        entity_text = entity_result.get("entity_text", "")
        entity_label = entity_result.get("entity_label", "")
        matched_nodes = entity_result.get("matched_nodes", [])
        
        if not matched_nodes:
            continue
        
        # 添加实体信息
        entity_section = f"实体：{entity_text}"
        if entity_label:
            entity_section += f"（类型：{entity_label}）"
        content_parts.append(entity_section)
        
        # 添加匹配的节点和关系信息
        for node in matched_nodes:
            node_name = node.get("name", "")
            similarity = node.get("similarity", 0)
            relations = node.get("relations", {})
            
            if node_name:
                node_info = f"  - {node_name}（相似度：{similarity:.3f}）"
                content_parts.append(node_info)
                
                # 添加关系信息
                if relations:
                    for rel_type, neighbors in relations.items():
                        neighbor_names = [n.get("name", "") for n in neighbors if n.get("name")]
                        if neighbor_names:
                            rel_info = f"    {rel_type}：{', '.join(neighbor_names)}"
                            content_parts.append(rel_info)
        
        content_parts.append("")  # 空行分隔
    
    return "\n".join(content_parts) if content_parts else "未找到相关知识图谱信息。"


def call_workflow(query: str, workflow_name: Optional[str] = None, conversation_id: Optional[str] = None, 
                  kg_content: Optional[str] = None) -> Tuple[Optional[dict], Optional[str]]:
    """
    Invoke the remote workflow/chat API and return JSON response or error message.
    
    Args:
        query: 用户问题（原始问题文本）
        workflow_name: 工作流名称
        conversation_id: 对话会话ID（用于多轮对话）
        kg_content: 知识图谱内容（可选，如果提供则直接使用，不进行 NER+KG 检索）
    """
    if workflow_name is None:
        workflow_name = get_default_workflow_name()
    
    workflow = WORKFLOWS.get(workflow_name)
    if workflow is None:
        return None, f"工作流 '{workflow_name}' 不存在"
    
    url = workflow['url']
    
    # 如果没有提供 kg_content，则使用空字符串（不进行检索，避免重复）
    if kg_content is None:
        kg_content = ""
    
    # 根据 URL 路径判断 API 类型并构造 payload
    if '/chat-messages' in url:
        # 聊天应用 API 格式
        payload = {
            "inputs": {
                "knowledge_graph_content": kg_content,  # 知识图谱内容
            },
            "query": query, 
            "response_mode": "blocking",  
            "user": USER_ID,
            "files": []
        }
        # 如果有会话 ID，添加到 payload（用于多轮对话）
        if conversation_id:
            payload["conversation_id"] = conversation_id
        # 如果有其他输入变量，合并到 inputs
        if workflow.get("inputs"):
            payload["inputs"].update(workflow["inputs"])

    elif '/workflows/run' in url:
        # 工作流 API 格式
        payload = {
            "inputs": {
                "query": query,  # 用户问题
                "knowledge_graph_content": kg_content,  # 知识图谱内容
            },
            "response_mode": "blocking",
            "user": USER_ID,
        }
        # 如果有其他输入变量，合并到 inputs
        if workflow.get("inputs"):
            payload["inputs"].update(workflow["inputs"])
    else:
        # 默认使用工作流格式
        payload = {
            "inputs": {
                "query": query,  # 用户问题
                "knowledge_graph_content": kg_content,  # 知识图谱内容
            },
            "response_mode": "blocking",
            "user": USER_ID,
        }
        # 如果有其他输入变量，合并到 inputs
        if workflow.get("inputs"):
            payload["inputs"].update(workflow["inputs"])
    
    headers = {
        "Authorization": f"Bearer {workflow['key']}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload, ensure_ascii=False), timeout=300)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = exc.response.text
        return None, f"接口调用失败：{exc}. {detail}"


def extract_display_text(result: dict) -> str:
    """Best-effort extraction of human-readable answer from workflow/chat API response."""
    if not isinstance(result, dict):
        return str(result)

    # 尝试多种响应格式
    # 1. 聊天应用 API 格式: data.answer
    data = result.get("data")
    if isinstance(data, dict):
        # 聊天应用返回的答案通常在 data.answer 中
        if isinstance(data.get("answer"), str):
            return data["answer"]
        
        # 工作流可能返回 outputs
        outputs = data.get("outputs")
        if isinstance(outputs, list) and outputs:
            # Dify workflow nodes typically return objects with a 'text' field.
            text_candidates = []
            for output in outputs:
                if isinstance(output, dict):
                    for key in ("text", "answer", "content", "data"):
                        value = output.get(key)
                        if isinstance(value, str):
                            text_candidates.append(value)
            if text_candidates:
                return "\n\n".join(text_candidates)
        
        if isinstance(data.get("result"), str):
            return data["result"]

    # 2. 顶层 answer 字段
    if isinstance(result.get("answer"), str):
        return result["answer"]
    
    # 3. 如果都找不到，返回格式化后的 JSON
    return json.dumps(result, ensure_ascii=False, indent=2)


def initialize_session_state():
    """初始化 session state"""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "chat_windows" not in st.session_state:
        st.session_state.chat_windows = [{"messages": [], "conversation_id": None}]
    if "active_window" not in st.session_state:
        st.session_state.active_window = 0
    if "selected_workflow" not in st.session_state:
        st.session_state.selected_workflow = get_default_workflow_name()


def login_page():
    """登录页面"""
    st.title("智能问答机器人")
    st.subheader("请登录您的账户")
    
    with st.form("login_form"):
        username = st.text_input("用户名", placeholder="请输入用户名")
        password = st.text_input("密码", type="password", placeholder="请输入密码")
        submit = st.form_submit_button("登录", use_container_width=True)
        
        if submit:
            if not username or not password:
                st.error("请输入用户名和密码")
            else:
                user_cred = credentials.get(username.strip())
                if user_cred and user_cred.password == password.strip():
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()
                    st.session_state.is_admin = user_cred.is_admin
                    st.rerun()
                else:
                    st.error("用户名或密码错误，请重新输入。")
    
    st.markdown("---")
    if st.button("还没有账户？立即注册", use_container_width=True):
        st.session_state.page = "register"
        st.rerun()


def register_page():
    """注册页面"""
    st.title("注册新账户")
    st.subheader("创建您的账户以开始使用")
    
    with st.form("register_form"):
        username = st.text_input("用户名", placeholder="请输入用户名")
        password = st.text_input("密码", type="password", placeholder="请输入密码")
        confirm_password = st.text_input("确认密码", type="password", placeholder="请再次输入密码")
        submit = st.form_submit_button("注册", use_container_width=True)
        
        if submit:
            username = username.strip()
            password = password.strip()
            confirm_password = confirm_password.strip()
            
            if not username or not password:
                st.error("用户名和密码不能为空。")
            # 已移除密码长度限制：原限制为至少6位，现允许任意长度
            # elif len(password) < 6:
            #     st.error("密码长度至少为6位")
            elif password != confirm_password:
                st.error("两次输入的密码不一致")
            elif username in credentials:
                st.error("用户名已存在，请使用其他用户名。")
            else:
                new_user = Credentials(username, password, is_admin=False)
                credentials[username] = new_user
                write_credentials(storage_file, credentials)
                st.success(f"用户 {username} 注册成功！请登录。")
                st.session_state.page = "login"
                st.rerun()
    
    st.markdown("---")
    if st.button("已有账户？立即登录", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()


def display_entities(entities):
    """显示实体识别结果"""
    if not entities:
        return
    
    with st.expander(f"实体识别结果 ({len(entities)} 个)", expanded=False):
        cols = st.columns(3)
        for idx, entity in enumerate(entities):
            with cols[idx % 3]:
                st.markdown(f"**{entity.get('text', '')}**")
                st.caption(f"类型: {entity.get('label', '')}")


def display_kg_results(kg_results):
    """显示知识图谱检索结果"""
    if not kg_results or not kg_results.get("entities"):
        return
    
    total_matched = kg_results.get("total_matched", 0)
    with st.expander(f"知识图谱检索结果 ({total_matched} 个实体匹配)", expanded=False):
        for entity_result in kg_results["entities"]:
            entity_text = entity_result.get("entity_text", "")
            entity_label = entity_result.get("entity_label", "")
            matched_nodes = entity_result.get("matched_nodes", [])
            
            if matched_nodes:
                st.markdown(f"**实体：{entity_text}** ({entity_label})")
                
                for node in matched_nodes:
                    node_name = node.get("name", "")
                    similarity = node.get("similarity", 0)
                    relations = node.get("relations", {})
                    
                    with st.container():
                        st.markdown(f"- **{node_name}** (相似度: {similarity:.3f})")
                        
                        if relations:
                            for rel_type, neighbors in relations.items():
                                neighbor_names = [n.get("name", "") for n in neighbors if n.get("name")]
                                if neighbor_names:
                                    st.markdown(f"  - {rel_type}: {', '.join(neighbor_names)}")
                
                st.markdown("---")


def main_page():
    """主页面"""
    # 侧边栏
    with st.sidebar:
        st.title("智能问答")
        st.caption("基于 Dify 工作流")
        
        # 用户信息
        user_type = "管理员" if st.session_state.is_admin else "用户"
        st.markdown(f"**欢迎您，{user_type} {st.session_state.username}**")
        st.caption("版本 1.0")
        
        st.markdown("---")
        
        # 对话窗口管理
        st.subheader("对话窗口")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("+ 新建窗口", use_container_width=True):
                st.session_state.chat_windows.append({"messages": [], "conversation_id": None})
                st.session_state.active_window = len(st.session_state.chat_windows) - 1
                st.rerun()
        
        with col2:
            if len(st.session_state.chat_windows) > 1:
                if st.button("删除窗口", use_container_width=True, type="secondary"):
                    if len(st.session_state.chat_windows) > 1:
                        st.session_state.chat_windows.pop(st.session_state.active_window)
                        if st.session_state.active_window >= len(st.session_state.chat_windows):
                            st.session_state.active_window = len(st.session_state.chat_windows) - 1
                        st.rerun()
        
        window_options = [f"对话窗口 {i+1}" for i in range(len(st.session_state.chat_windows))]
        selected_window = st.selectbox(
            "选择窗口",
            window_options,
            index=st.session_state.active_window
        )
        if selected_window != window_options[st.session_state.active_window]:
            st.session_state.active_window = window_options.index(selected_window)
            st.rerun()
        
        st.markdown("---")
        
        # 工作流选择
        st.subheader("工作流选择")
        workflow_names = list(WORKFLOWS.keys())
        selected_workflow = st.selectbox(
            "选择工作流",
            workflow_names,
            index=workflow_names.index(st.session_state.selected_workflow) if st.session_state.selected_workflow in workflow_names else 0
        )
        if selected_workflow != st.session_state.selected_workflow:
            st.session_state.selected_workflow = selected_workflow
        
        # 管理员选项
        if st.session_state.is_admin:
            st.markdown("---")
            st.subheader("管理员选项")
            st.session_state.show_raw_response = st.checkbox("显示原始响应", value=st.session_state.get("show_raw_response", False))
        
        st.markdown("---")
        
        # 操作按钮
        if st.button("清空当前对话", use_container_width=True, type="secondary"):
            st.session_state.chat_windows[st.session_state.active_window]["messages"] = []
            st.session_state.chat_windows[st.session_state.active_window]["conversation_id"] = None
            st.rerun()
        
        if st.button("退出登录", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.is_admin = False
            st.session_state.chat_windows = [{"messages": [], "conversation_id": None}]
            st.session_state.active_window = 0
            st.rerun()
    
    # 主内容区
    st.title("智能问答机器人")
    
    # 获取当前窗口的消息
    current_window = st.session_state.chat_windows[st.session_state.active_window]
    messages = current_window.get("messages", [])
    
    # 显示消息历史
    if messages:
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                with st.chat_message("user"):
                    st.write(content)
            else:
                with st.chat_message("assistant"):
                    st.write(content)
                    
                    # 显示实体识别结果
                    if msg.get("entities"):
                        display_entities(msg["entities"])
                    
                    # 显示知识图谱检索结果
                    if msg.get("kg_results"):
                        display_kg_results(msg["kg_results"])
                    
                    # 管理员可以看到工作流和原始响应
                    if st.session_state.is_admin:
                        st.caption(f"工作流: {msg.get('workflow', 'N/A')}")
                        
                        if st.session_state.get("show_raw_response") and msg.get("raw_response"):
                            with st.expander("查看原始响应（管理员）", expanded=False):
                                st.code(json.dumps(msg["raw_response"], ensure_ascii=False, indent=2), language="json")
    else:
        st.info("👋 欢迎使用智能问答机器人！请输入您的问题，我会尽力为您解答。")
    
    # 输入区域
    query = st.chat_input("输入您的问题...")
    
    if query:
        # 确保用户消息内容就是纯粹的query，去除首尾空白，避免包含其他内容
        user_message_content = query.strip()
        
        # 添加用户消息到消息列表（保存纯粹的用户输入内容）
        current_window["messages"].append({"role": "user", "content": user_message_content})
        
        # 立即显示用户消息（让用户看到自己的输入）
        with st.chat_message("user"):
            st.write(user_message_content)
        
        # 显示加载提示
        with st.chat_message("assistant"):
            with st.spinner("正在思考中..."):
                # Step 1: NER 实体识别
                # 注意：使用user_message_content而不是query，确保使用的是纯粹的用户输入
                entities = []
                try:
                    ner_model = get_ner_model()
                    entities = ner_model.extract_entities(user_message_content)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    # NER 失败不影响后续流程，继续执行
                
                # Step 2: 知识图谱检索
                kg_results = {}
                if entities:
                    try:
                        top_k = app_config.KG_TOP_K
                        similarity_threshold = app_config.KG_SIMILARITY_THRESHOLD
                        
                        # 获取rerank配置
                        enable_rerank = app_config.KG_ENABLE_RERANK
                        rerank_top_n = app_config.KG_RERANK_TOP_N
                        rerank_threshold = app_config.KG_RERANK_THRESHOLD
                        
                        kg_retriever = get_kg_retriever()
                        kg_results = kg_retriever.retrieve_knowledge(
                            entities, 
                            top_k=top_k, 
                            similarity_threshold=similarity_threshold,
                            enable_rerank=enable_rerank,
                            query=user_message_content,  # 传入纯粹的用户查询用于rerank，避免包含机器人回复
                            rerank_top_n=rerank_top_n,
                            rerank_threshold=rerank_threshold
                        )
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        # KG 检索失败不影响后续流程，继续执行
                
                # Step 3: 格式化知识图谱内容
                kg_content = format_kg_content(kg_results) if kg_results else ""
                
                # Step 4: 调用 Dify 工作流/聊天 API
                # 注意：使用user_message_content而不是query，确保传递的是纯粹的用户输入，不会包含机器人回复
                conversation_id = current_window.get("conversation_id")
                result, err = call_workflow(
                    user_message_content, 
                    st.session_state.selected_workflow, 
                    conversation_id, 
                    kg_content=kg_content
                )
                
                if err:
                    answer = f"错误：{err}"
                else:
                    # 从响应中提取 conversation_id
                    if isinstance(result, dict):
                        new_conversation_id = (
                            result.get("conversation_id") or 
                            result.get("data", {}).get("conversation_id") or
                            conversation_id
                        )
                        if new_conversation_id:
                            current_window["conversation_id"] = new_conversation_id
                    
                    answer = extract_display_text(result)
                    
                    # 保存助手回复
                    assistant_msg = {
                        "role": "assistant",
                        "content": answer,
                        "workflow": st.session_state.selected_workflow,
                        "entities": entities,
                        "kg_results": kg_results,
                    }
                    
                    # 管理员可以看到原始响应
                    if st.session_state.is_admin:
                        assistant_msg["raw_response"] = result
                    
                    current_window["messages"].append(assistant_msg)
                    
                    # 注意：不在这里立即显示助手回复，而是通过 st.rerun() 后由上面的消息历史循环统一显示
                    # 这样可以确保消息按正确顺序显示，避免多轮对话时消息混乱和重复显示的问题
                    
        # 重新渲染页面，让消息历史循环统一显示所有消息（包括刚添加的用户消息和助手回复）
        # 这样可以避免多轮对话时消息显示混乱的问题
        st.rerun()


def main():
    """主函数"""
    # 设置页面配置
    st.set_page_config(
        page_title="智能问答机器人",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化 session state
    initialize_session_state()
    
    # 根据登录状态显示不同页面
    if not st.session_state.logged_in:
        if st.session_state.get("page") == "register":
            register_page()
        else:
            login_page()
    else:
        main_page()


if __name__ == "__main__":
    main()

