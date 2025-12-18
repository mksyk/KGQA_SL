"""
NER 模块：使用 Qwen2.5-MedNER 模型进行医疗实体识别
"""
import torch
import json
import re
import logging
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NERModel:
    def __init__(self, base_model_path: str, checkpoint_path: str, device: str = "cuda"):
        """
        初始化 NER 模型
        
        Args:
            base_model_path: 基础模型路径
            checkpoint_path: LoRA checkpoint 路径
            device: 设备类型
        """
        self.device = device
        logger.info(f"正在加载基础模型: {base_model_path}")
        
        # 加载基础模型和 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, 
            use_fast=False, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        
        # 加载 LoRA 权重
        logger.info(f"正在加载 LoRA 权重: {checkpoint_path}")
        self.model = PeftModel.from_pretrained(self.model, model_id=checkpoint_path)
        self.model.eval()
        
        logger.info("NER 模型加载完成")
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        从医疗文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表，每个实体包含 text 和 label
        """
        instruction = """你是一个医疗文本实体识别领域的专家，你需要从给定的医疗文本中提取以下类型的实体：疾病、身体部位、症状、治疗方法、检查项目、药品。以 json 格式输出，如 {"entity_text": "肺炎", "entity_label": "疾病"} 注意: 1. 输出的每一行都必须是正确的 json 字符串，多个实体用逗号分隔。2. 找不到任何实体时, 输出"没有找到任何实体"。"""
        
        input_value = f"文本:{text}"
        
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_value}
        ]
        
        # 生成回复
        text_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids, 
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False
            )
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 解析响应，提取实体
        entities = self._parse_entities(response)
        return entities
    
    def _parse_entities(self, response: str) -> List[Dict[str, str]]:
        """
        解析模型输出，提取实体
        新格式：多个实体用逗号分隔，如 {"entity_text": "肺炎", "entity_label": "疾病"}, {"entity_text": "发热", "entity_label": "症状"}
        参照 test.py 的实现方式
        
        Args:
            response: 模型输出文本
            
        Returns:
            实体列表，每个实体包含 text 和 label
        """
        entities = []
        response = response.strip()
        
        # 检查是否没有找到实体
        if "没有找到任何实体" in response:
            return entities
        
        # 移除可能的 markdown 代码块标记
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # 方法1：尝试将响应包装成 JSON 数组格式（参照 test.py 的实现）
        try:
            # 如果响应已经是数组格式
            if response.startswith('[') and response.endswith(']'):
                entity_list = json.loads(response)
            else:
                # 将响应包装成数组格式
                json_str = f'[{response}]'
                entity_list = json.loads(json_str)
            
            # 解析实体列表
            for item in entity_list:
                if isinstance(item, dict):
                    entity_text = item.get("entity_text") or item.get("text")
                    entity_label = item.get("entity_label") or item.get("label")
                    
                    if entity_text and entity_label:
                        entities.append({
                            "text": entity_text,
                            "label": entity_label
                        })
            return entities
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"方法1（JSON数组）解析失败: {e}, 尝试方法2")
        
        # 方法2：如果 JSON 解析失败，尝试使用正则表达式逐个匹配 JSON 对象
        try:
            # 使用更精确的正则表达式匹配 JSON 对象（支持嵌套引号）
            # 匹配格式: {"entity_text": "...", "entity_label": "..."}
            json_pattern = r'\{\s*"entity_text"\s*:\s*"([^"]+)"\s*,\s*"entity_label"\s*:\s*"([^"]+)"\s*\}'
            matches = re.finditer(json_pattern, response)
            
            for match in matches:
                entity_text = match.group(1)
                entity_label = match.group(2)
                entities.append({
                    "text": entity_text,
                    "label": entity_label
                })
            
            if entities:
                return entities
        except Exception as e:
            logger.debug(f"方法2（正则表达式）解析失败: {e}, 尝试方法3")
        
        # 方法3：尝试查找所有可能的 JSON 对象并逐个解析
        try:
            # 查找所有 { } 包围的内容
            brace_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            potential_json_objects = re.findall(brace_pattern, response)
            
            for obj_str in potential_json_objects:
                try:
                    entity = json.loads(obj_str)
                    entity_text = entity.get("entity_text")
                    entity_label = entity.get("entity_label")
                    
                    if entity_text and entity_label:
                        entities.append({
                            "text": entity_text,
                            "label": entity_label
                        })
                except json.JSONDecodeError:
                    continue
            
            if entities:
                return entities
        except Exception as e:
            logger.warning(f"所有解析方法都失败: {e}, 响应内容: {response}")
        
        return entities


# 全局 NER 模型实例（延迟加载）
_ner_model = None


def get_ner_model(base_model_path: str = None,
                  checkpoint_path: str = None,
                  device: str = None) -> NERModel:
    """获取 NER 模型实例（单例模式）"""
    global _ner_model
    if _ner_model is None:
        # 从配置文件读取参数
        try:
            import config as app_config
            base_model_path = base_model_path or app_config.NER_BASE_MODEL_PATH
            checkpoint_path = checkpoint_path or app_config.NER_CHECKPOINT_PATH
            device = device or app_config.NER_DEVICE
        except ImportError:
            # 如果配置文件不存在，使用默认值（新模型路径）
            base_model_path = base_model_path or "/home/lx/crj/KG4medLLM/models/qwen2.5-1.5B-instruct"
            checkpoint_path = checkpoint_path or "/home/lx/crj/KG4medLLM/output/Qwen2.5-MedNER/checkpoint-1688"
            device = device or "cuda"
        
        _ner_model = NERModel(base_model_path, checkpoint_path, device)
    return _ner_model

