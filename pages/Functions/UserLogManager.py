import os
import json
from datetime import datetime
import re


class UserLogManager:
    def __init__(self, base_path="user_logs"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _sanitize_name(self, username):
        # 仅保留数字和大小写字母，移除其他所有字符
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', username)
        return sanitized[:50].strip()

    def user_register(self, username):
        safe_username = self._sanitize_name(username)
        user_path = os.path.join(self.base_path, safe_username)
        # 防止路径遍历
        user_path = os.path.normpath(user_path)
        if not user_path.startswith(os.path.abspath(self.base_path)):
            raise ValueError("非法用户名")
        os.makedirs(os.path.join(self.base_path, safe_username), exist_ok=True)

    def _get_user_path(self, username):
        safe_username = self._sanitize_name(username)
        return os.path.join(self.base_path, safe_username)

    def check_user_exists(self, username):
        """检查用户是否存在"""
        return os.path.exists(self._get_user_path(username))

    def save_chat_log(self, username, messages, log_filename=None):
        """保存对话记录"""
        user_dir = self._get_user_path(username)
        os.makedirs(user_dir, exist_ok=True)

        if log_filename:
            file_path = os.path.join(user_dir, log_filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                merged_messages = existing_data["messages"] + messages
                messages = merged_messages[-40:]
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            first_user_content = "无用户输入"
            for msg in messages:
                if msg["role"] == "user":
                    first_user_content = msg["content"]
                    break
            safe_filename = self._sanitize_name(first_user_content)
            filename = f"{safe_filename}_{timestamp}.json"
            file_path = os.path.join(user_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "username": username,
                "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
                "messages": messages
            }, f, ensure_ascii=False, indent=2)

        return os.path.basename(file_path)

    def get_user_history(self, username):
        """获取用户历史记录列表"""
        user_dir = self._get_user_path(username)
        if not os.path.exists(user_dir):
            return []

        logs = sorted(os.listdir(user_dir), reverse=True)
        return [log.rsplit(".json", 1)[0] for log in logs if log.endswith('.json')]

    def load_chat_log(self, username, log_filename):
        """加载特定聊天记录"""
        log_filename += '.json'
        user_dir = self._get_user_path(username)
        file_path = os.path.join(user_dir, log_filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def delete_chat_log(self, username, log_filename):
        """删除指定聊天记录"""
        user_dir = self._get_user_path(username)
        file_path = os.path.join(user_dir, log_filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    def get_log_filepath(self, username, log_filename):
        """获取日志文件的完整路径"""
        user_dir = self._get_user_path(username)
        return os.path.join(user_dir, log_filename)
