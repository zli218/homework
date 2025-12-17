import ast
from typing import List, Dict, Any

class CodeParser:
    def parse_code(self, file_content: str, file_name: str) -> List[Dict[str, Any]]:
        """
        解析代码，提取 Class 和 Function 定义。
        返回结构化数据列表。
        """
        chunks = []
        try:
            tree = ast.parse(file_content)
            lines = file_content.splitlines()

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # 获取源码片段
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    code_segment = "\n".join(lines[start_line:end_line])
                    
                    # 提取 Docstring
                    docstring = ast.get_docstring(node) or "No docstring provided."
                    
                    chunks.append({
                        "type": "class" if isinstance(node, ast.ClassDef) else "function",
                        "name": node.name,
                        "file_name": file_name,
                        "code": code_segment,
                        "docstring": docstring
                    })
        except SyntaxError:
            pass # 忽略解析错误的文件
            
        return chunks
