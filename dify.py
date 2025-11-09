"""
title: DIFY Manifold Pipe
authors: erlyhuang
version: 0.1.2
description: 该流程用于DIFY的API接口，用于与DIFY的API进行交互
"""

import os
import requests
import json
import time
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
from open_webui.config import UPLOAD_DIR
import base64
import tempfile


def get_file_extension(file_name: str) -> str:
    """
    获取文件的扩展名
    os.path.splitext(file_name) 返回一个元组, 第一个元素是文件名, 第二个元素是扩展名
    """
    return os.path.splitext(file_name)[1].strip(".")


# 黑魔法, 从__event_emitter__中获取闭包变量
def get_closure_info(func):
    # 获取函数的闭包变量
    if hasattr(func, "__closure__") and func.__closure__:
        for cell in func.__closure__:
            if isinstance(cell.cell_contents, dict):
                return cell.cell_contents
    return None


class Pipe:
    class Valves(BaseModel):
        # 环境变量的设置
        DIFY_BASE_URL: str = Field(default="http://xxxx:xxxx/v1")
        DIFY_KEY: str = Field(default="xxx")
        FILE_SERVER: str = Field(default="http://xxxx:xxxx")

    def __init__(self):
        self.type = "manifold"
        self.id = "baojun"
        self.name = "baojun/"

        self.valves = self.Valves(**{"DIFY_KEY": os.getenv("DIFY_KEY", "")})

        # 存储格式:
        # {
        #   "chat_id_1": {
        #     "dify_conversation_id": "xxx",
        #     "messages": [{"chat_message_id_1": "dify_message_id_1"}, ...]
        #   }
        # }
        self.chat_message_mapping = {}

        # 存储格式:
        # {
        #   "chat_id_1": "gpt-3.5-turbo",
        #   "chat_id_2": "gpt-4"
        # }
        self.dify_chat_model = {}

        # 存储格式:
        # {
        #   "chat_id_1": {
        #     "file_id1":{
        #       "local_file_path": "/path/to/file1.pdf",
        #       "dify_file_id": "dify_file_123",
        #       "file_name": "file1.pdf"
        #     },
        #     "file_id2":{
        #       "local_file_path": "/path/to/file2.jpg",
        #       "dify_file_id": "dify_file_456",
        #       "file_name": "file2.jpg"
        #     }
        #   }
        # }
        self.dify_file_list = {}

        self.data_cache_dir = "data/dify"
        self.load_state()

    def save_state(self):
        """持久化Dify相关的状态变量到文件"""
        os.makedirs(self.data_cache_dir, exist_ok=True)

        # chat_message_mapping.json
        chat_mapping_file = os.path.join(
            self.data_cache_dir, "chat_message_mapping.json"
        )
        with open(chat_mapping_file, "w", encoding="utf-8") as f:
            json.dump(self.chat_message_mapping, f, ensure_ascii=False, indent=2)

        # chat_model.json
        chat_model_file = os.path.join(self.data_cache_dir, "chat_model.json")
        with open(chat_model_file, "w", encoding="utf-8") as f:
            json.dump(self.dify_chat_model, f, ensure_ascii=False, indent=2)

        # file_list.json
        file_list_file = os.path.join(self.data_cache_dir, "file_list.json")
        with open(file_list_file, "w", encoding="utf-8") as f:
            json.dump(self.dify_file_list, f, ensure_ascii=False, indent=2)

    def load_state(self):
        """从文件加载Dify相关的状态变量"""
        try:
            # chat_message_mapping.json
            chat_mapping_file = os.path.join(
                self.data_cache_dir, "chat_message_mapping.json"
            )
            if os.path.exists(chat_mapping_file):
                with open(chat_mapping_file, "r", encoding="utf-8") as f:
                    self.chat_message_mapping = json.load(f)
            else:
                self.chat_message_mapping = {}

            # chat_model.json
            chat_model_file = os.path.join(self.data_cache_dir, "chat_model.json")
            if os.path.exists(chat_model_file):
                with open(chat_model_file, "r", encoding="utf-8") as f:
                    self.dify_chat_model = json.load(f)
            else:
                self.dify_chat_model = {}

            # file_list.json
            file_list_file = os.path.join(self.data_cache_dir, "file_list.json")
            if os.path.exists(file_list_file):
                with open(file_list_file, "r", encoding="utf-8") as f:
                    self.dify_file_list = json.load(f)
            else:
                self.dify_file_list = {}

        except Exception as e:
            print(f"加载Dify状态文件失败: {e}")
            # 加载失败时使用空字典
            self.chat_message_mapping = {}
            self.dify_chat_model = {}
            self.dify_file_list = {}

    def get_models(self):
        """
        获取DIFY的模型列表
        """

        return [{"id": "study", "name": "study"}]

    def upload_file(self, user_id: str, file_path: str, mime_type: str) -> str:
        """
        这个函数负责上传文件到DIFY的服务器, 并返回文件的ID
        """
        import requests

        url = f"{self.valves.DIFY_BASE_URL}/files/upload"
        headers = {
            "Authorization": f"Bearer {self.valves.DIFY_KEY}",
        }

        file_name = os.path.basename(file_path)

        files = {
            # 文件字段：(文件名, 文件对象, MIME类型)
            "file": (file_name, open(file_path, "rb"), mime_type),
            # 普通表单字段：(None, 值)
            "user": (None, user_id),
        }

        response = requests.post(url, headers=headers, files=files)

        file_id = response.json()["id"]

        # Optional: print response
        return file_id

    def is_doc_file(self, file_path: str) -> bool:
        """
        判断文件是否是文档文件
        'TXT', 'MD', 'MARKDOWN', 'PDF', 'HTML', 'XLSX', 'XLS', 'DOCX', 'CSV', 'EML', 'MSG', 'PPTX', 'PPT', 'XML', 'EPUB'
        """
        file_extension = get_file_extension(file_path).upper()
        if file_extension in [
            "PDF",
            "XLSX",
            "XLS",
            "DOCX",
            "EML",
            "MSG",
            "PPTX",
            "PPT",
            "XML",
            "EPUB",
        ]:
            return True

        return False

    def is_text_file(self, mime_type: str) -> bool:
        """
        判断文件是否是文本文件
        """
        if "text" in mime_type:
            return True
        return False

    def is_audio_file(self, file_path: str) -> bool:
        """
        判断文件是否是音频文件
        'MP3', 'M4A', 'WAV', 'WEBM', 'AMR'.
        """
        if get_file_extension(file_path).upper() in [
            "MP3",
            "M4A",
            "WAV",
            "WEBM",
            "AMR",
        ]:
            return True
        return False

    def is_video_file(self, file_path: str) -> bool:
        """
        判断文件是否是视频文件
        'MP4', 'MOV', 'MPEG', 'MPGA'
        """
        if get_file_extension(file_path).upper() in ["MP4", "MOV", "MPEG", "MPGA"]:
            return True
        return False

    def upload_text_file(self, user_id: str, file_path: str) -> str:
        """
        上传文本文件到服务器，第一行添加文件名
        支持类型: 纯文本文件

        Args:
            file_path: 文本文件路径
            user_id: 用户ID

        Returns:
            str: 上传后的文件ID
        """
        try:
            # 获取原始文件名
            filename = os.path.basename(file_path)

            # 读取原文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 创建带文件名标记的新内容
            new_content = f"#{filename}\n{content}"

            # 创建临时文件
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt", mode="w", encoding="utf-8"
            ) as tmp_file:
                tmp_file.write(new_content)
                temp_file_path = tmp_file.name

            try:
                # 上传文件
                file_id = self.upload_file(user_id, temp_file_path, "text/plain")
                print("file_id")
                print(file_id)
                return file_id
            finally:
                # 清理临时文件
                os.remove(temp_file_path)

        except UnicodeDecodeError:
            raise ValueError("文件编码不是UTF-8格式")
        except Exception as e:
            raise ValueError(f"处理文本文件失败: {str(e)}")

    def upload_images(self, image_data_base64: str, user_id: str) -> str:
        """
        上传 base64 编码的图片到 DIFY 服务器，返回图片路径
        支持类型: 'JPG', 'JPEG', 'PNG', 'GIF', 'WEBP', 'SVG'
        """
        try:
            # Remove the data URL scheme prefix if present
            if image_data_base64.startswith("data:"):
                # Extract the base64 data after the comma
                image_data_base64 = image_data_base64.split(",", 1)[1]

            # 解码 base64 图像数据
            image_data = base64.b64decode(image_data_base64)

            # Create and save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(image_data)
                temp_file_path = tmp_file.name
            try:
                file_id = self.upload_file(user_id, temp_file_path, "image/png")
                print("file_id")
                print(file_id)
            finally:
                os.remove(temp_file_path)

            return file_id
        except Exception as e:
            raise ValueError(f"Failed to process base64 image data: {str(e)}")

    def pipes(self) -> List[dict]:
        return self.get_models()

    def pipe(
        self,
        body: dict,
        __event_emitter__: dict = None,
        __user__: Optional[dict] = None,
        __task__=None,
    ) -> Union[str, Generator, Iterator]:
        """处理对话请求"""

        # 获取模型名称
        model_name = body["model"][body["model"].find(".") + 1 :]
        # 处理特殊任务
        if __task__ is not None:
            if __task__ == "title_generation":
                return model_name
            elif __task__ == "tags_generation":
                return f'{{"tags":[{model_name}]}}'

        # 获取当前用户
        current_user = __user__["email"]

        # 处理系统消息和普通消息
        system_message, messages = pop_system_message(body["messages"])
        # print(f"system_message:{system_message}")
        # print(f"messages:{messages}, {len(messages)}")

        # 从event_emitter获取chat_id和message_id
        cell_contents = get_closure_info(__event_emitter__)
        chat_id = cell_contents["chat_id"]
        message_id = cell_contents["message_id"]
        print(f"chat_id:{chat_id}")
        print(f"message_id:{message_id}")
        # 处理对话模型和上下文
        parent_message_id = None
        # 在pipe函数中修改对话历史的处理逻辑
        print(f"len(messages) {len(messages)}")
        if len(messages) == 1:
            # 新对话逻辑保持不变
            print("===========新对话逻辑保持不变================")
            self.dify_chat_model[chat_id] = model_name
            self.chat_message_mapping[chat_id] = {
                "dify_conversation_id": "",
                "messages": [],
            }

            self.dify_file_list[chat_id] = {}
        else:
            # 检查是否存在历史记录
            if chat_id in self.chat_message_mapping:
                # print(f"检查是否存在历史记录 chat_id {chat_id} {self.dify_chat_model}")
                # 首先验证模型
                if chat_id in self.dify_chat_model:
                    if self.dify_chat_model[chat_id] != model_name:
                        raise ValueError(
                            f"Cannot change model in an existing conversation. This conversation was started with {self.dify_chat_model[chat_id]}"
                        )
                else:
                    # 如果somehow没有记录模型（异常情况），记录当前模型
                    self.dify_chat_model[chat_id] = model_name
                # print(f"self.chat_message_mapping{self.chat_message_mapping}")
                chat_history = self.chat_message_mapping[chat_id]["messages"]
                # current_msg_index = len(messages) - 1  # 当前消息的索引
                current_msg_index = int((len(messages) - 1) / 2)
                print(f"current_msg_index:{current_msg_index}")
                print(f"chat_history：{chat_history}")
                print(f"len(chat_history) {len(chat_history)}")
                # 如果不是第一条消息，获取前一条消息的dify_id作为parent
                if current_msg_index > 0 and len(chat_history) >= current_msg_index:
                    print(
                        "=======如果不是第一条消息，获取前一条消息的dify_id作为parent========"
                    )

                    previous_msg = chat_history[current_msg_index - 1]
                    parent_message_id = list(previous_msg.values())[0]
                    print(f"parent_message_id:{parent_message_id}")
                    # 关键修改：截断当前位置之后的消息历史
                    self.chat_message_mapping[chat_id]["messages"] = chat_history[
                        :current_msg_index
                    ]
        print(
            f"=====获取最后一条消息作为query======self.chat_message_mapping[chat_id] {self.chat_message_mapping[chat_id]}"
        )
        # 获取最后一条消息作为query
        message = messages[-1]
        query = ""
        # inputs = {"model": model_name}
        inputs = {}
        file_list = []

        # 处理消息内容
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item["type"] == "text":
                    query += item["text"]
                if item["type"] == "image_url":
                    upload_file_id = self.upload_images(
                        item["image_url"]["url"], current_user
                    )
                    print(f"upload_file_id:{upload_file_id}")
                    upload_file_dict = {
                        "type": "image",
                        "transfer_method": "local_file",
                        "url": "",
                        "upload_file_id": upload_file_id,
                    }
                    inputs["file"] = {
                        "type": "image",
                        "transfer_method": "local_file",
                        "url": "",
                        "upload_file_id": upload_file_id,
                    }

                    file_list.append(upload_file_dict)
        else:
            query = message.get("content", "")

        # 处理文件上传
        if "upload_files" in body:
            for file in body["upload_files"]:
                if file["type"] != "file":
                    continue

                file_id = file["id"]
                if (
                    chat_id in self.dify_file_list
                    and file_id in self.dify_file_list[chat_id]
                ):
                    file_list.append(self.dify_file_list[chat_id][file_id])
                    continue

                # 获取文件信息并上传
                if "collection_name" in file:
                    file_path = os.path.join(UPLOAD_DIR, file["file"]["filename"])
                else:
                    file_path = file["file"]["path"]
                file_mime_type = file["file"]["meta"]["content_type"]

                upload_file_dict = {
                    "transfer_method": "local_file",
                    "url": "",
                }

                # 处理不同类型的文件
                if self.is_doc_file(file_path):
                    upload_file_id = self.upload_file(
                        current_user, file_path, file_mime_type
                    )
                    upload_file_dict.update(
                        {"type": "document", "upload_file_id": upload_file_id}
                    )
                elif self.is_text_file(file_mime_type):
                    upload_file_id = self.upload_text_file(current_user, file_path)
                    upload_file_dict.update(
                        {"type": "document", "upload_file_id": upload_file_id}
                    )
                elif self.is_audio_file(file_path):
                    upload_file_id = self.upload_file(
                        current_user, file_path, file_mime_type
                    )
                    upload_file_dict.update(
                        {"type": "audio", "upload_file_id": upload_file_id}
                    )
                elif self.is_video_file(file_path):
                    upload_file_id = self.upload_file(
                        current_user, file_path, file_mime_type
                    )
                    upload_file_dict.update(
                        {"type": "video", "upload_file_id": upload_file_id}
                    )
                else:
                    raise ValueError(f"Unsupported file type: {file_path}")

                file_list.append(upload_file_dict)
                if chat_id not in self.dify_file_list:
                    self.dify_file_list[chat_id] = {}
                self.dify_file_list[chat_id][file_id] = upload_file_dict

        print(
            f"self.chat_message_mapping[chat_id] {self.chat_message_mapping[chat_id]}"
        )
        conversation_id = self.chat_message_mapping[chat_id].get(
            "dify_conversation_id", ""
        )
        print(f"conversation_id {conversation_id}")
        # 准备请求载荷
        payload = {
            "inputs": inputs,
            "parent_message_id": parent_message_id,
            "query": query,
            "response_mode": "streaming" if body.get("stream", False) else "blocking",
            "conversation_id": self.chat_message_mapping[chat_id].get(
                "dify_conversation_id", ""
            ),
            "user": current_user,
            "files": file_list,
        }

        print(f"======准备请求载荷payload {payload}")

        # 设置请求头
        headers = {
            "Authorization": f"Bearer {self.valves.DIFY_KEY}",
            "content-type": "application/json",
        }

        url = f"{self.valves.DIFY_BASE_URL}/chat-messages"

        try:
            if body.get("stream", False):
                return self.stream_response(url, headers, payload, chat_id, message_id)
            else:
                return self.non_stream_response(
                    url, headers, payload, chat_id, message_id
                )
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload, chat_id, message_id):
        """处理流式响应"""
        print(f"======处理流式响应======payload {payload}")
        try:
            with requests.post(
                # url, headers=headers, json=payload, stream=True, timeout=(3.05, 60)
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(60, 300),
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                event = data.get("event")

                                if event == "message":
                                    # 处理普通文本消息
                                    yield data.get("answer", "")
                                elif event == "message_file":
                                    # 处理文件（图片）消息
                                    pass
                                elif event == "message_end":
                                    # 保存会话和消息ID映射
                                    dify_conversation_id = data.get(
                                        "conversation_id", ""
                                    )
                                    dify_message_id = data.get("message_id", "")
                                    print(
                                        "=====保存会话和消息ID映射=chat_message_mapping===前===="
                                    )
                                    print(self.chat_message_mapping[chat_id])
                                    self.chat_message_mapping[chat_id][
                                        "dify_conversation_id"
                                    ] = dify_conversation_id
                                    self.chat_message_mapping[chat_id][
                                        "messages"
                                    ].append({message_id: dify_message_id})
                                    print(
                                        "=====保存会话和消息ID映射=chat_message_mapping===后===="
                                    )
                                    print(self.chat_message_mapping[chat_id])
                                    # 保存状态
                                    self.save_state()
                                    break
                                elif event == "error":
                                    # 处理错误
                                    error_msg = f"Error {data.get('status')}: {data.get('message')} ({data.get('code')})"
                                    yield f"Error: {error_msg}"
                                    break

                                time.sleep(0.01)
                            except json.JSONDecodeError:
                                print(f"Failed to parse JSON: {line}")
                            except KeyError as e:
                                print(f"Unexpected data structure: {e}")
                                print(f"Full data: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            yield f"Error: Request failed: {e}"
        except Exception as e:
            print(f"General error in stream_response method: {e}")
            yield f"Error: {e}"

    def non_stream_response(self, url, headers, payload, chat_id, message_id):
        """处理非流式响应"""
        try:
            print(f"处理非流式响应 ======payload ======== {payload}")
            response = requests.post(
                url, headers=headers, json=payload, timeout=(60, 300)
            )
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            res = response.json()
            print("res")
            print(res)
            # 保存会话和消息ID映射
            dify_conversation_id = res.get("conversation_id", "")
            dify_mesage_id = res.get("message_id", "")
            print("dify_mesage_id")
            print(dify_mesage_id)
            print("dify_conversation_id")
            print(dify_conversation_id)
            self.chat_message_mapping[chat_id][
                "dify_conversation_id"
            ] = dify_conversation_id
            self.chat_message_mapping[chat_id]["messages"].append(
                {message_id: dify_mesage_id}
            )

            # 保存状态
            self.save_state()

            return res.get("answer", "")
        except requests.exceptions.RequestException as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"
