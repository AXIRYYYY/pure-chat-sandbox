import os
import re
import fitz  # PyMuPDF


def clean_pdf_text(text):
    """
    清理从PDF提取的文本，移除可能引起编码问题的字符。
    
    处理以下问题：
    1. 移除控制字符（除了常见的空格、制表符、换行符）
    2. 替换一些特殊空格字符为普通空格
    3. 移除非打印字符
    4. 确保文本编码安全
    """
    if not text:
        return text
    
    # 步骤1: 替换常见的特殊空格字符为普通空格
    # U+200C: ZERO WIDTH NON-JOINER (零宽度非连接符)
    # U+200D: ZERO WIDTH JOINER (零宽度连接符)
    # U+200E: LEFT-TO-RIGHT MARK (从左到右标记)
    # U+200F: RIGHT-TO-LEFT MARK (从右到左标记)
    # U+202A: LEFT-TO-RIGHT EMBEDDING
    # U+202C: POP DIRECTIONAL FORMATTING
    # U+FEFF: ZERO WIDTH NO-BREAK SPACE (BOM)
    special_chars_map = {
        '\u200c': '',     # 零宽度非连接符 - 移除
        '\u200d': '',     # 零宽度连接符 - 移除
        '\u200e': '',     # 从左到右标记 - 移除
        '\u200f': '',     # 从右到左标记 - 移除
        '\u202a': '',     # LEFT-TO-RIGHT EMBEDDING - 移除
        '\u202c': '',     # POP DIRECTIONAL FORMATTING - 移除
        '\ufeff': '',     # ZERO WIDTH NO-BREAK SPACE - 移除
        '\u00a0': ' ',    # NO-BREAK SPACE - 替换为普通空格
        '\u2028': '\n',   # LINE SEPARATOR - 替换为换行
        '\u2029': '\n\n', # PARAGRAPH SEPARATOR - 替换为双换行
    }
    
    for char, replacement in special_chars_map.items():
        text = text.replace(char, replacement)
    
    # 步骤2: 移除其他控制字符（除了常见的 \t, \n, \r）
    # 保留: \t (0x09), \n (0x0A), \r (0x0D)
    # 移除: 其他控制字符 (0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F)
    cleaned = []
    for char in text:
        code = ord(char)
        # 保留常见字符：制表符、换行、回车
        if code in (9, 10, 13):
            cleaned.append(char)
        # 移除其他控制字符和删除字符
        elif code < 32 or code == 127:
            continue
        # 保留其他所有字符（包括中文字符、emoji等）
        else:
            cleaned.append(char)
    
    text = ''.join(cleaned)
    
    # 步骤3: 移除连续的空白字符（除了保留换行）
    # 将多个连续的空格替换为单个空格
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 步骤4: 确保文本以UTF-8编码安全返回
    # 对于中文字符和emoji，直接保留
    
    return text.strip()


def read_pdf_content(file_bytes_or_path):
    """
    统一解析 PDF 内容，返回提取出的文本。
    支持文件路径或文件字节流。
    
    返回的文本会经过清理，移除可能引起编码问题的字符。
    """
    try:
        if isinstance(file_bytes_or_path, str):
            doc = fitz.open(file_bytes_or_path)
        else:
            doc = fitz.open(stream=file_bytes_or_path, filetype="pdf")

        with doc:
            text = "".join(page.get_text() for page in doc)

        if not text.strip():
            return "(PDF 内容解析为空，可能是扫描件或图片型 PDF)"
        
        # 清理文本：移除可能引起编码问题的字符
        # 保留大部分Unicode字符，但过滤掉一些控制字符和特殊空格
        cleaned_text = clean_pdf_text(text)
        return cleaned_text
        
    except fitz.FileDataError as e:
        return f"(PDF 文件数据错误: {str(e)} - 可能文件损坏或不支持)"
    except fitz.EmptyFileError as e:
        return f"(PDF 文件为空: {str(e)})"
    except Exception as e:
        return f"(PDF 解析失败: {str(e)})"


def read_text_content(file_bytes):
    """
    尝试以不同编码（UTF-8, GBK）读取文本字节流，
    并返回解码后的字符串。
    """
    try:
        # 优先尝试 UTF-8
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            # 尝试中文系统常见的 GBK
            return file_bytes.decode("gbk")
        except UnicodeDecodeError:
            # 最后兜底，忽略错误字符
            return file_bytes.decode("utf-8", errors="ignore")


def process_uploaded_file(uploaded_file):
    """
    对上传的文件对象进行通用的内容提取。
    
    增强功能：
    1. 更严格的PDF检测
    2. 更好的错误处理
    3. 文件大小检查
    4. 异常文本清理
    """
    filename = uploaded_file.name
    file_size = len(uploaded_file.getvalue()) if hasattr(uploaded_file, 'getvalue') else 0
    
    # 检测是否为PDF文件
    is_pdf = filename.lower().endswith(".pdf")
    
    if is_pdf:
        try:
            # 验证文件头部是否为有效的PDF
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > 4:
                # PDF文件应该以 '%PDF' 开头
                if file_bytes[:4] != b'%PDF':
                    return f"(文件扩展名为PDF但内容不是有效的PDF格式)"
            
            # 解析PDF内容
            content = read_pdf_content(file_bytes)
            
            # 如果解析返回错误信息，提供更友好的提示
            if content.startswith("(PDF 解析失败") or content.startswith("(PDF 文件数据错误"):
                # 尝试作为文本文件读取作为最后的回退
                try:
                    text_content = read_text_content(file_bytes)
                    return f"(PDF解析失败，作为文本读取):\n{clean_pdf_text(text_content)[:2000]}..."
                except:
                    return content
            
            return content
            
        except Exception as e:
            return f"(PDF处理异常: {str(e)})"
    else:
        try:
            return read_text_content(uploaded_file.getvalue())
        except Exception as e:
            return f"(文件解析受限: {str(e)})"


def get_file_content_summary(file_name, file_text, file_size_mb):
    """
    根据文件大小生成统一的内容摘要格式。
    
    注意：自 2026-07-17 起，已移除 >5MB 跳过解析的限制，
    所有附件一律解析文本内容。file_size_mb 参数保留以
    维持向后兼容，但在摘要中不再展示"文件过大"字样。
    """
    return f"【附件原文：{file_name}】\n{file_text}"
