import os
import git
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RepoLoader:
    def __init__(self, repo_url: str, local_path: str):
        self.repo_url = repo_url
        self.local_path = local_path

    def clone_repo(self):
        """克隆或更新代码仓"""
        if os.path.exists(self.local_path):
            logger.info(f"仓库已存在于 {self.local_path}，跳过克隆。")
            return
        
        logger.info(f"正在克隆 {self.repo_url} ...")
        try:
            git.Repo.clone_from(self.repo_url, self.local_path)
            logger.info("克隆完成。")
        except git.exc.GitCommandNotFound:
            logger.error("错误：未找到 git 命令。请安装 Git (https://git-scm.com/) 并确保其在 PATH 环境变量中。")
            raise

    def get_python_files(self) -> List[str]:
        """获取所有 Python 文件的路径"""
        py_files = []
        for root, _, files in os.walk(self.local_path):
            for file in files:
                if file.endswith(".py") and "test" not in file: # 简单过滤掉测试文件，聚焦业务逻辑
                    py_files.append(os.path.join(root, file))
        return py_files

    def read_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
