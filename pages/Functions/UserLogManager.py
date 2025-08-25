import os
import json
from datetime import datetime
import re
import zipfile
import io
import aiofiles
import aiofiles.os as aios


class UserLogManager:
    def __init__(self, base_path="user_logs"):
        self.base_path = base_path
        self._initialized = False

    async def ensure_initialized(self):
        """确保目录已初始化"""
        if not self._initialized:
            await aios.makedirs(self.base_path, exist_ok=True)
            self._initialized = True

    async def _sanitize_name(self, username):
        # 保留数字、大小写字母和汉字，移除其他所有字符
        if username is None:
            return ""
        sanitized = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '', username)
        return sanitized[:50].strip()

    async def user_register(self, username):
        await self.ensure_initialized()
        safe_username = await self._sanitize_name(username)
        await aios.makedirs(os.path.join(self.base_path, safe_username), exist_ok=True)

    async def _get_user_path(self, username):
        await self.ensure_initialized()
        safe_username = await self._sanitize_name(username)
        return os.path.join(self.base_path, safe_username)

    async def check_user_exists(self, username):
        """检查用户是否存在"""
        return await aios.path.exists(await self._get_user_path(username))

    async def save_chat_log(self, username, messages, log_filename=None):
        """保存对话记录"""
        user_dir = await self._get_user_path(username)
        await aios.makedirs(user_dir, exist_ok=True)

        if log_filename:
            file_path = os.path.join(user_dir, log_filename)
            if await aios.path.exists(file_path):
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    existing_data = json.loads(content)
                merged_messages = existing_data["messages"] + messages
                messages = merged_messages[-40:]
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            first_user_content = "无用户输入"
            for msg in messages:
                if msg["role"] == "user":
                    first_user_content = msg["content"]
                    break
            safe_filename = await self._sanitize_name(first_user_content)
            filename = f"{safe_filename}_{timestamp}.json"
            file_path = os.path.join(user_dir, filename)

        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps({
                "username": username,
                "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
                "messages": messages
            }, ensure_ascii=False, indent=2))

        return os.path.basename(file_path)

    async def get_user_history(self, username):
        """获取用户历史记录列表"""
        user_dir = await self._get_user_path(username)
        if not await aios.path.exists(user_dir):
            return []

        files = await aios.listdir(user_dir)
        logs = sorted([f for f in files if f.endswith('.json')], reverse=True)
        return [log.rsplit(".json", 1)[0] for log in logs]

    async def load_chat_log(self, username, log_filename):
        """加载特定聊天记录"""
        log_filename += '.json'
        user_dir = await self._get_user_path(username)
        file_path = os.path.join(user_dir, log_filename)

        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content)

    async def delete_chat_log(self, username, log_filename):
        """删除指定聊天记录"""
        user_dir = await self._get_user_path(username)
        file_path = os.path.join(user_dir, log_filename)
        if await aios.path.exists(file_path):
            await aios.remove(file_path)
            return True
        return False

    async def get_log_filepath(self, username, log_filename):
        """获取日志文件的完整路径"""
        user_dir = await self._get_user_path(username)
        return os.path.join(user_dir, log_filename)


class KnowledgeBaseManager(UserLogManager):
    def __init__(self, base_path="user_knowledge"):
        super().__init__(base_path)
        self.knowledge_dir = base_path
        self._knowledge_initialized = False

    async def ensure_knowledge_initialized(self):
        """确保知识库目录已初始化"""
        if not self._knowledge_initialized:
            await self.ensure_initialized()
            await aios.makedirs(self.knowledge_dir, exist_ok=True)
            self._knowledge_initialized = True
    
    async def _get_user_knowledge_path(self, username):
        """获取用户知识库目录路径"""
        await self.ensure_knowledge_initialized()
        safe_username = await self._sanitize_name(username)
        user_knowledge_dir = os.path.join(self.knowledge_dir, safe_username)
        await aios.makedirs(user_knowledge_dir, exist_ok=True)
        return user_knowledge_dir
    
    async def save_knowledge_base(self, username, file_id, data):
        """保存用户知识库"""
        user_knowledge_dir = await self._get_user_knowledge_path(username)
        json_path = os.path.join(user_knowledge_dir, f"knowledge_{file_id}.json")
        
        async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        
        return json_path
    
    async def get_knowledge_base(self, username, file_id):
        """获取用户知识库"""
        user_knowledge_dir = await self._get_user_knowledge_path(username)
        json_path = os.path.join(user_knowledge_dir, f"knowledge_{file_id}.json")
        
        if await aios.path.exists(json_path):
            async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        return None
    
    async def list_knowledge_bases(self, username):
        """列出用户的所有知识库"""
        user_knowledge_dir = await self._get_user_knowledge_path(username)
        if not await aios.path.exists(user_knowledge_dir):
            return []
        
        files = await aios.listdir(user_knowledge_dir)
        knowledge_files = [f for f in files 
                          if f.startswith("knowledge_") and f.endswith(".json")]
        
        knowledge_bases = []
        for file in knowledge_files:
            file_id = file.replace("knowledge_", "").replace(".json", "")
            file_path = os.path.join(user_knowledge_dir, file)

            file_stat = await aios.stat(file_path)
            created_time = datetime.fromtimestamp(file_stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
            
            # 获取知识库内容摘要
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
                chunk_count = len(data)
                # 获取第一段文本的前50个字符作为摘要
                summary = data[0]["text"][:50] + "..." if data else "空知识库"
            
            knowledge_bases.append({
                "file_id": file_id,
                "created_time": created_time,
                "chunk_count": chunk_count,
                "summary": summary,
                "file_path": file_path
            })
        
        return knowledge_bases
    
    async def delete_knowledge_base(self, username, file_id):
        """删除用户知识库"""
        user_knowledge_dir = await self._get_user_knowledge_path(username)
        json_path = os.path.join(user_knowledge_dir, f"knowledge_{file_id}.json")
        
        if await aios.path.exists(json_path):
            await aios.remove(json_path)
            return True
        return False
    
    async def download_knowledge_base(self, username, file_id):
        """下载用户知识库"""
        user_knowledge_dir = await self._get_user_knowledge_path(username)
        json_path = os.path.join(user_knowledge_dir, f"knowledge_{file_id}.json")
        
        if await aios.path.exists(json_path):
            async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(f"knowledge_{file_id}.json", 
                                 json.dumps(data, ensure_ascii=False, indent=2))

                metadata = {
                    "username": username,
                    "file_id": file_id,
                    "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "chunk_count": len(data)
                }
                zip_file.writestr("metadata.json", 
                                 json.dumps(metadata, ensure_ascii=False, indent=2))
            
            zip_buffer.seek(0)
            return zip_buffer
        return None
    
    async def download_all_knowledge_bases(self, username):
        """下载用户所有知识库"""
        user_knowledge_dir = await self._get_user_knowledge_path(username)
        if not await aios.path.exists(user_knowledge_dir):
            return None
        
        files = await aios.listdir(user_knowledge_dir)
        knowledge_files = [f for f in files 
                          if f.startswith("knowledge_") and f.endswith(".json")]
        
        if not knowledge_files:
            return None

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in knowledge_files:
                file_path = os.path.join(user_knowledge_dir, file)
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
                zip_file.writestr(file, json.dumps(data, ensure_ascii=False, indent=2))

            metadata = {
                "username": username,
                "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "knowledge_base_count": len(knowledge_files)
            }
            zip_file.writestr("metadata.json", 
                             json.dumps(metadata, ensure_ascii=False, indent=2))
        
        zip_buffer.seek(0)
        return zip_buffer
