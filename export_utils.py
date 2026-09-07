def export_chat_to_md(chat_title, messages):
    """
    将对话导出为 Markdown 格式，仅包含正文、附件/文件和时间戳。
    """
    md_content = f"# {chat_title}\n\n---\n\n"

    for msg in messages:
        role = msg.get("role", "").upper()
        md_content += f"### [{role}]\n\n"

        # 提取附件 (兼容新版 attachments 列表和旧版 attachment 字符串)
        attachments = msg.get("attachments", [])
        if not attachments and msg.get("attachment"):
            attachments = [msg.get("attachment")]

        if attachments:
            md_content += f"**关联资产:** `{', '.join(attachments)}` \n\n"

        # 提取内容和时间戳
        content = msg.get("content", "")
        timestamp_str = ""

        if role == "USER":
            # 尝试从 user 的 content 中剥离出时间戳
            if "\n\n[提交于：" in content:
                parts = content.rsplit("\n\n[提交于：", 1)
                content = parts[0]
                time_part = parts[1].replace("]", "").strip()
                timestamp_str = time_part
        elif role == "ASSISTANT":
            # 从 assistant 的元数据中提取时间戳
            raw_time = msg.get("timestamp", "")
            if raw_time and raw_time != "N/A":
                # 转换 ISO 格式为 YYYY-MM-DD HH:MM:SS
                timestamp_str = raw_time[:19].replace("T", " ")

        md_content += f"{content}\n\n"

        if timestamp_str:
            md_content += f"*(时间: {timestamp_str})*\n\n"

        md_content += "---\n\n"

    return md_content
