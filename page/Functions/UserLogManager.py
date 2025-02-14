import os
import json
from datetime import datetime
import re


class UserLogManager:
    def __init__(self, base_path="user_logs"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def user_register(self, username):
        os.makedirs(os.path.join(self.base_path, username), exist_ok=True)

    def _get_user_path(self, username):
        return os.path.join(self.base_path, username)

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
            safe_filename = re.sub(r'[\\/*?:"<>|]', "_", first_user_content)
            safe_filename = safe_filename[:50].strip()
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
