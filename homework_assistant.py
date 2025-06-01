# homework_assistant.py
import os
import json
import re
import uuid
import argparse
import time
import traceback
import base64
from pathlib import Path
from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model
from qwen_agent.tools import BaseTool
import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

# 环境变量配置
ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ALIYUN_API_BASE = os.getenv("ALIYUN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# 日志输出函数
def log(message):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

# 图像处理函数
def encode_image_to_base64(image_path):
    """将图像文件编码为Base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        log(f"图像编码失败: {str(e)}")
        return None

# ===================== 自定义工具 =====================
class ImageAnalysisTool(BaseTool):
    name = 'image_analysis'
    description = '识别图片中的错题内容'
    
    def call(self, params: dict):
        return self._run(params['image_path'], params['subject'])
    
    def _run(self, image_path: str, subject: str):
        if not os.path.exists(image_path):
            return f"错误：文件 {image_path} 不存在"
        
        # 添加重试机制
        for retry in range(3):
            try:
                # 将图像转换为Base64
                image_base64 = encode_image_to_base64(image_path)
                if not image_base64:
                    return f"错误：图像编码失败"
                
                # 构建请求
                llm_cfg = {
                    'model': 'qwen-vl-plus', 
                    'model_server': ALIYUN_API_BASE, 
                    'api_key': ALIYUN_API_KEY,
                    'stream': True
                }
                
                # 修正：直接使用base64内容而不是data URI格式
                messages = [{
                    'role': 'user',
                    'content': [
                        {
                            'image': image_base64,  # 关键修复：直接使用base64
                        },
                        {'text': f'请提取这张{subject}错题图片中的题目内容'}
                    ]
                }]
                
                # 获取完整的响应
                log(f"使用Qwen-VL模型分析图像: {image_path}")
                gen = get_chat_model(llm_cfg).chat(messages)
                
                # 处理流式响应
                response_text = ""
                for chunk in gen:
                    if isinstance(chunk, dict):
                        if 'content' in chunk:
                            response_text += chunk['content']
                        elif 'text' in chunk:
                            response_text += chunk['text']
                    else:
                        response_text += str(chunk)
                
                log(f"图像识别结果: {response_text[:100]}...")
                return response_text
                    
            except Exception as e:
                error_msg = f"图片识别失败: {str(e)}\n{traceback.format_exc()}"
                log(error_msg)
                if retry < 2:
                    log(f"图片识别重试中 (重试 {retry+1}/3)...")
                    time.sleep(2)
                else:
                    return error_msg
        
        return "图片识别失败: 重试次数已用完"


class ErrorAnalysisTool(BaseTool):
    name = 'error_analysis'
    description = '分析错题原因和知识点'
    
    def call(self, params: dict):
        return self._run(params['question_text'], params['subject'])
    
    def _run(self, question_text: str, subject: str):
        try:
            llm_cfg = {
                'model': 'deepseek-reasoner', 
                'model_server': DEEPSEEK_API_BASE, 
                'api_key': DEEPSEEK_API_KEY,
                'stream': True
            }
            prompt = f"""作为小学{subject}教师，分析以下错题：
【题目】
{question_text}

请返回JSON格式分析结果：
error_reason: 错误原因
knowledge_point: 涉及知识点
knowledge_analysis: 知识点解析
correct_approach: 正确解题思路"""
            
            log(f"使用DeepSeek模型分析错题: {subject}")
            gen = get_chat_model(llm_cfg).chat([{'role': 'user', 'content': prompt}])
            
            # 处理流式响应
            response_text = ""
            for chunk in gen:
                if isinstance(chunk, dict):
                    if 'content' in chunk:
                        response_text += chunk['content']
                    elif 'text' in chunk:
                        response_text += chunk['text']
                else:
                    response_text += str(chunk)
            
            log(f"错题分析原始响应: {response_text[:200]}")
            
            # 尝试解析JSON
            try:
                match = re.search(r'\{[\s\S]*\}', response_text)
                if match:
                    json_content = match.group(0)
                    result = json.loads(json_content)
                    log(f"解析JSON成功: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    return result
                else:
                    log(f"JSON格式未找到，尝试直接加载整个内容")
                    return json.loads(response_text)
            except Exception as e:
                log(f"JSON解析失败: {str(e)}\n尝试解析的内容: {response_text[:200]}")
                return {"error": f"JSON解析失败: {str(e)}", "raw_content": response_text}
            
        except Exception as e:
            error_msg = f"分析失败: {str(e)}\n{traceback.format_exc()}"
            log(error_msg)
            return {"error": error_msg}


class QuestionGeneratorTool(BaseTool):
    name = 'question_generator'
    description = '生成练习题'
    
    def call(self, params: dict):
        return self._run(params['analysis_result'], params['subject'])
    
    def _run(self, analysis_result: dict, subject: str):
        try:
            if not isinstance(analysis_result, dict) or analysis_result.get('error'):
                return {"error": "无效的分析结果"}
                
            llm_cfg = {
                'model': 'deepseek-chat', 
                'model_server': DEEPSEEK_API_BASE, 
                'api_key': DEEPSEEK_API_KEY,
                'stream': True
            }
            knowledge_point = analysis_result.get('knowledge_point', '')
            prompt = f"""根据知识点: {knowledge_point}
生成3道小学{subject}练习题。返回JSON格式: {{"questions": [{{"question": "题目", "answer": "答案"}}]}}"""
            
            log(f"使用DeepSeek模型生成练习题")
            gen = get_chat_model(llm_cfg).chat([{'role': 'user', 'content': prompt}])
            
            # 处理流式响应
            response_text = ""
            for chunk in gen:
                if isinstance(chunk, dict):
                    if 'content' in chunk:
                        response_text += chunk['content']
                    elif 'text' in chunk:
                        response_text += chunk['text']
                else:
                    response_text += str(chunk)
            
            log(f"练习题生成原始响应: {response_text[:200]}")
            
            # 尝试解析JSON
            try:
                match = re.search(r'\{[\s\S]*\}', response_text)
                if match:
                    json_content = match.group(0)
                    result = json.loads(json_content)
                    log(f"解析JSON成功: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    return result
                else:
                    log(f"JSON格式未找到，尝试直接加载整个内容")
                    return json.loads(response_text)
            except Exception as e:
                log(f"JSON解析失败: {str(e)}\n尝试解析的内容: {response_text[:200]}")
                return {"error": f"JSON解析失败: {str(e)}", "raw_content": response_text}
            
        except Exception as e:
            error_msg = f"生成失败: {str(e)}\n{traceback.format_exc()}"
            log(error_msg)
            return {"error": error_msg}


class ReportGeneratorTool(BaseTool):
    name = 'report_generator'
    description = '生成报告'
    
    def call(self, params: dict):
        report_dir = params.get('report_dir', 'reports')
        return self._run(
            params['image_path'],
            params['question_text'],
            params['analysis'],
            params['practice_questions'],
            report_dir
        )
    
    def _run(self, image_path: str, question_text: str, analysis: dict, 
             practice_questions, report_dir: str = "reports"):
        try:
            # 确保报告目录存在
            Path(report_dir).mkdir(parents=True, exist_ok=True)
            
            # 生成唯一的图片文件名
            img_ext = Path(image_path).suffix
            img_name = f"原题_{uuid.uuid4().hex[:6]}{img_ext}"
            img_path = Path(report_dir) / img_name
            
            # 复制图片到报告目录
            with open(image_path, 'rb') as src, open(img_path, 'wb') as dst:
                dst.write(src.read())
            
            # 生成报告文件
            report_path = Path(report_dir) / f"错题报告_{uuid.uuid4().hex[:6]}.md"
            
            # 构建报告内容
            content = self._generate_markdown(img_name, question_text, analysis, practice_questions)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return str(report_path)
            
        except Exception as e:
            error_msg = f"报告失败: {str(e)}\n{traceback.format_exc()}"
            log(error_msg)
            return error_msg
    
    def _generate_markdown(self, img_name, question_text, analysis, practice_questions):
        """生成Markdown格式的报告内容"""
        # 确保question_text是字符串
        if not isinstance(question_text, str):
            question_text = str(question_text)
        
        # 处理分析部分
        def get_analysis_value(key, default="未提供"):
            if isinstance(analysis, dict):
                return analysis.get(key, default)
            return default
        
        # 处理练习部分
        practice_section = "## 🧠 巩固练习\n"
        if practice_questions:
            if isinstance(practice_questions, dict) and practice_questions.get('questions'):
                for i, q in enumerate(practice_questions['questions'][:3]):
                    question = q.get('question', '未知问题')
                    answer = q.get('answer', '未知答案')
                    practice_section += f"""### 练习题 {i+1}
{question}

<details>
<summary>查看答案</summary>

{answer}
</details>
"""
            elif isinstance(practice_questions, dict) and 'error' in practice_questions:
                error_msg = practice_questions.get('error', '未知错误')
                practice_section += f"⚠️ 练习题目生成失败: {error_msg}"
            else:
                practice_section += f"⚠️ 练习题目生成失败: 响应类型为 {type(practice_questions)}"
        else:
            practice_section += "⚠️ 练习题目生成失败: 无结果返回"
        
        # 完整的报告
        return f"""# 📝 错题分析报告

## 📷 原始题目
![]({img_name})

## 📝 题目内容
{question_text}

## 错误分析
### ❌ 错误原因
{get_analysis_value('error_reason')}

### 📚 知识点
{get_analysis_value('knowledge_point')}

### 📖 知识点解析
{get_analysis_value('knowledge_analysis')}

### ✅ 正确解题思路
{get_analysis_value('correct_approach')}

{practice_section}

---
*生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""


# ===================== 主调度Agent =====================
class HomeworkAssistant(Assistant):
    def __init__(self):
        tools = [
            ImageAnalysisTool(),
            ErrorAnalysisTool(),
            QuestionGeneratorTool(),
            ReportGeneratorTool()
        ]
        super().__init__(
            function_list=tools,
            llm={'model': 'qwen-max', 'model_server': ALIYUN_API_BASE, 'api_key': ALIYUN_API_KEY}
        )
        
        # 工具调用映射
        self.function_map = {tool.name: tool for tool in tools}
        
    def _call_tool(self, tool_name, params):
        """包装工具调用，处理异常并日志输出"""
        try:
            log(f"调用工具: {tool_name}")
            return getattr(self.function_map[tool_name], 'call')(params)
        except Exception as e:
            error_msg = f"工具 {tool_name} 调用失败: {str(e)}"
            log(error_msg)
            traceback.print_exc()
            return {"error": error_msg}
    
    def analyze_homework(self, image_path: str, subject: str = 'math'):
        """处理错题的核心方法"""
        log(f"开始处理{subject}错题: {image_path}")
        
        # 1. 图像识别
        log("识别图片内容...")
        question_text = self._call_tool('image_analysis', {'image_path': image_path, 'subject': subject})
        log(f"图像识别结果类型: {type(question_text)}\n内容: {str(question_text)[:100]}")

        # 检查识别结果
        if (isinstance(question_text, dict) and question_text.get('error')) or "失败" in str(question_text) or "错误" in str(question_text) or "抱歉" in str(question_text):
            return {"status": "error", "message": f"图片识别失败: {question_text}"}

        # 2. 错题分析
        log("分析错题...")
        analysis = self._call_tool('error_analysis', {'question_text': question_text, 'subject': subject})
        log(f"错题分析结果类型: {type(analysis)}\n内容: {str(analysis)[:200]}")

        # 检查分析结果
        if (isinstance(analysis, dict) and analysis.get('error')):
            return {"status": "error", "message": f"分析失败: {analysis.get('error')}"}
        
        # 3. 生成练习题
        log("生成练习题...")
        practice_questions = self._call_tool('question_generator', {'analysis_result': analysis, 'subject': subject})
        log(f"练习题目生成结果类型: {type(practice_questions)}\n内容: {str(practice_questions)[:200]}")
        
        # 检查练习题生成
        if (isinstance(practice_questions, dict) and practice_questions.get('error')):
            return {"status": "error", "message": f"生成失败: {practice_questions.get('error')}"}
        
        # 4. 生成报告
        log("生成报告...")
        report_path = self._call_tool('report_generator', {
            'image_path': image_path,
            'question_text': question_text,
            'analysis': analysis,
            'practice_questions': practice_questions
        })
        log(f"报告生成结果类型: {type(report_path)}\n内容: {str(report_path)}")
        
        # 检查报告生成
        if (isinstance(report_path, dict) and report_path.get('error')) or "失败" in str(report_path) or "错误" in str(report_path):
            return {"status": "error", "message": f"报告生成失败: {report_path}"}
        
        return {"status": "success", "report_path": report_path}


# ===================== 主执行函数 =====================
def main():
    parser = argparse.ArgumentParser(description='错题分析助手')
    parser.add_argument('image', help='错题图片路径')
    parser.add_argument('-s', '--subject', default='math', 
                       choices=['math', 'chinese', 'english'], help='学科类型')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.image):
        print(f"错误：文件不存在 {args.image}")
        return
    
    # 初始化智能体
    assistant = HomeworkAssistant()
    
    # 处理错题
    result = assistant.analyze_homework(args.image, args.subject)
    
    # 输出结果
    if result['status'] == 'success':
        print(f"✅ 完成! 报告路径: {result['report_path']}")
        print(f"✨ 你可以用以下命令查看报告: \n  cat '{result['report_path']}'")
    else:
        print(f"❌ 失败: {result['message']}")


if __name__ == '__main__':
    main()