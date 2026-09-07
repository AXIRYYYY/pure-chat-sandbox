import streamlit as st
import tiktoken
from google import genai
from google.genai import types
from provider_registry import is_gemini_type, is_openai_type, get_api_key as _pk


def normalize_gemini_model(model_id):
    """
    规范化Gemini模型ID，确保以models/开头但不会重复添加前缀。

    参数:
        model_id: 模型ID字符串，可能是短格式（如'gemini-2.5-pro'）
                  或完整格式（如'models/gemini-2.5-pro'）

    返回:
        规范化后的模型ID，确保以'models/'开头且不重复
    """
    if not model_id:
        return model_id
    if model_id.startswith("models/"):
        return model_id
    return f"models/{model_id}"


def calculate_prompt_tokens(
    messages, system_prompt, model_choice, target_model, api_configs
):
    """
    根据当前消息历史和选择的模型，通过官方 API 或 tiktoken 估算下一次请求将消耗的 Token 数量。
    """
    if not _pk(api_configs, model_choice):
        return "0 (估算)"
    if not target_model:
        return "0 (估算)"

    try:
        if is_gemini_type(model_choice):
            try:
                # --- 优先尝试官方精确计算接口 ---
                client = genai.Client(api_key=_pk(api_configs, model_choice))

                history_for_calc = []
                last_role = None
                for m in messages:
                    role = "user" if m["role"] == "user" else "model"
                    text_content = m.get("full_payload", m["content"])
                    if not text_content or not text_content.strip():
                        text_content = " "

                    # 补充用户初始消息（如果第一条是模型消息）
                    if not history_for_calc and role == "model":
                        history_for_calc.append(
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text("[System Init]")],
                            )
                        )
                        last_role = "user"

                    # 合并同角色的连续消息（SDK要求交替出现）
                    if role == last_role and history_for_calc:
                        combined_text = (
                            history_for_calc[-1].parts[0].text + f"\n\n{text_content}"
                        )
                        history_for_calc[-1].parts = [
                            types.Part.from_text(text=combined_text)
                        ]
                    else:
                        history_for_calc.append(
                            types.Content(
                                role=role,
                                parts=[types.Part.from_text(text=text_content)],
                            )
                        )
                        last_role = role

                model_to_use = normalize_gemini_model(target_model)
                params = {"model": model_to_use, "contents": history_for_calc}
                if system_prompt and system_prompt.strip():
                    params["system_instruction"] = system_prompt

                response = client.models.count_tokens(**params)
                return f"{response.total_tokens} (精确)"

            except Exception as official_e:
                # --- 如果官方 API 失败，自动回退到本地 tiktoken 估算 ---
                print(
                    f"Official Gemini token count failed ({official_e}), falling back to tiktoken."
                )
                encoding = tiktoken.get_encoding("cl100k_base")
                full_text = ""
                if system_prompt and system_prompt.strip():
                    full_text += f"system: {system_prompt}\n"
                for message in messages:
                    full_text += f"{message['role']}: {message.get('full_payload', message['content'])}\n"
                num_tokens = len(encoding.encode(full_text))
                return f"~{num_tokens} (估算)"

        elif is_openai_type(model_choice):
            # OpenAI 兼容通道统一使用 tiktoken 进行通用估算
            encoding = tiktoken.get_encoding("cl100k_base")
            full_text = ""
            if system_prompt and system_prompt.strip():
                full_text += f"system: {system_prompt}\n"
            for message in messages:
                full_text += f"{message['role']}: {message.get('full_payload', message['content'])}\n"
            num_tokens = len(encoding.encode(full_text))
            return f"~{num_tokens} (估算)"

    except Exception as e:
        print(f"Token calculation error: {e}")
        return "计算失败"


def update_token_estimate():
    """
    Streamlit 按钮的回调函数，用于触发计算并更新 session_state 中的显示结果。
    """
    st.session_state.estimated_tokens_display = calculate_prompt_tokens(
        st.session_state.messages,
        st.session_state.api_configs.get("system_prompt", ""),
        st.session_state.api_configs.get("last_choice", "Gemini"),
        st.session_state.api_configs.get("last_model", ""),
        st.session_state.api_configs,
    )
