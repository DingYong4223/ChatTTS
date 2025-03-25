import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from ChatTTS import Chat
from tools.logger import get_logger

def save_speaker_embedding(speaker_embedding, save_path, logger):
    """保存说话人音色到文件"""
    # 打印speaker_embedding的类型，用于调试
    logger.info(f"speaker_embedding type: {type(speaker_embedding)}")
    
    # 如果是字符串，直接保存
    if isinstance(speaker_embedding, str):
        with open(save_path, 'w') as f:
            f.write(speaker_embedding)
        logger.info(f"已保存音色到: {save_path}")
    else:
        logger.error(f"Unexpected speaker_embedding type: {type(speaker_embedding)}")
        raise TypeError(f"Unexpected speaker_embedding type: {type(speaker_embedding)}")

def test_generate_and_save_speaker():
    """测试用例：生成随机音色并保存到文件"""
    # 设置日志
    logger = get_logger("test_generate_speaker")
    
    # 初始化ChatTTS
    chat = Chat(logger)
    
    # 加载模型
    chat.load(source="local")
    
    # 创建输出目录
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 生成随机音色并保存
    speaker_embedding = chat.speaker.sample_random()
    speaker_save_path = output_dir / "speaker_embedding.txt"
    save_speaker_embedding(speaker_embedding, speaker_save_path, logger)
    
    # 验证文件是否存在和大小
    assert speaker_save_path.exists(), f"Speaker embedding file {speaker_save_path} was not created"
    assert speaker_save_path.stat().st_size > 0, "Speaker embedding file is empty"
    
    logger.info(f"Test completed successfully. Speaker embedding saved to {speaker_save_path}")
    return speaker_save_path

if __name__ == "__main__":
    test_generate_and_save_speaker() 