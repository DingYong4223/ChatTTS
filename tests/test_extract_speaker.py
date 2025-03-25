import os
import sys
import logging
from pathlib import Path
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ChatTTS.core import Chat
from tools.audio import load_audio
from tools.logger import get_logger

def load_speaker_embedding(file_path, logger):
    """加载音色文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            speaker_embedding = f.read().strip()
        logger.info(f"已从 {file_path} 加载音色")
        return speaker_embedding
    except Exception as e:
        logger.error(f"加载音色失败: {e}")
        raise

def test_extract_speaker():
    """测试从音频中提取音色"""
    # 设置日志
    logger = get_logger("test_extract_speaker")
    
    # 初始化 ChatTTS
    chat = Chat(logger)
    chat.load(source="local")
    
    # 创建输出目录
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 加载原始音色
    original_speaker_path = output_dir / "speaker_embedding.txt"
    original_speaker_embedding = load_speaker_embedding(original_speaker_path, logger)
    
    # 加载音频文件
    audio_path = output_dir / "speaker_load_test_cn.mp3"
    if not audio_path.exists():
        logger.error(f"音频文件不存在: {audio_path}")
        return
    
    # 加载并处理音频
    audio = load_audio(audio_path, sr=24000, mono=True)
    logger.info(f"音频数据形状: {audio.shape}")
    logger.info(f"音频数据范围: [{audio.min():.3f}, {audio.max():.3f}]")
    logger.info(f"音频数据均值: {audio.mean():.3f}")
    logger.info(f"音频数据标准差: {audio.std():.3f}")
    
    if len(audio.shape) > 1:
        audio = audio.mean(axis=0)
    
    # 从音频中提取音色
    extracted_speaker_embedding = chat.sample_audio_speaker(audio)
    logger.info(f"提取的音色类型: {type(extracted_speaker_embedding)}")
    logger.info(f"提取的音色长度: {len(extracted_speaker_embedding)}")
    
    # 保存提取的音色
    extracted_speaker_path = output_dir / "extracted_speaker_embedding.txt"
    with open(extracted_speaker_path, 'w', encoding='utf-8') as f:
        f.write(extracted_speaker_embedding)
    logger.info(f"已保存提取的音色到: {extracted_speaker_path}")
    
    # 比较原始音色和提取的音色
    if original_speaker_embedding == extracted_speaker_embedding:
        logger.info("音色提取成功：提取的音色与原始音色完全一致")
    else:
        logger.warning("音色提取可能存在问题：提取的音色与原始音色不一致")
        logger.info(f"原始音色长度: {len(original_speaker_embedding)}")
        logger.info(f"提取音色长度: {len(extracted_speaker_embedding)}")
        # 计算相似度
        common_chars = set(original_speaker_embedding) & set(extracted_speaker_embedding)
        similarity = len(common_chars) / max(len(original_speaker_embedding), len(extracted_speaker_embedding))
        logger.info(f"音色相似度: {similarity:.2%}")

if __name__ == "__main__":
    test_extract_speaker() 