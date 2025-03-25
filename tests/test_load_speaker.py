import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from ChatTTS import Chat
from tools.logger import get_logger
from tools.audio import pcm_arr_to_mp3_view

def load_speaker_embedding(load_path, logger):
    """从文件加载说话人音色"""
    with open(load_path, 'r') as f:
        speaker_embedding = f.read()
    logger.info(f"已从 {load_path} 加载音色")
    return speaker_embedding

def test_load_speaker_and_generate_audio():
    """测试用例：从文件加载音色并生成测试音频"""
    # 设置日志
    logger = get_logger("test_load_speaker")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 初始化ChatTTS
    chat = Chat(logger)
    
    # 加载模型
    chat.load(source="local")
    
    # 创建输出目录
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 测试文本
    test_text = "你好，这是一个测试音频。Hello, this is a test audio."
    
    # 从文件加载音色
    speaker_save_path = output_dir / "speaker_embedding.txt"
    loaded_speaker_embedding = load_speaker_embedding(speaker_save_path, logger)
    
    # 设置生成参数
    params = chat.InferCodeParams(
        spk_emb=loaded_speaker_embedding,
        manual_seed=42,  # 固定随机种子
        temperature=0.1,  # 降低温度参数，使生成更稳定
        top_P=0.9,  # 提高top_P值，使采样更集中
        top_K=40,  # 增加top_K值，扩大采样范围
        repetition_penalty=1.0  # 降低重复惩罚
    )
    
    # 使用加载的音色生成语音
    output_path = output_dir / "speaker_load_test.mp3"
    wavs = chat.infer(
        text=test_text,
        params_infer_code=params,
        split_text=False,  # 禁用文本分割
        do_text_normalization=True,  # 保持文本标准化
        do_homophone_replacement=True  # 保持同音字替换
    )
    
    # 保存音频文件
    if isinstance(wavs, list):
        wavs = wavs[0]  # 如果返回多个音频，取第一个
    try:
        data = pcm_arr_to_mp3_view(wavs)
        with open(output_path, "wb") as f:
            f.write(data)
        logger.info(f"已保存音频文件: {output_path}")
    except Exception as e:
        logger.error(f"保存音频文件失败: {e}")
        raise
    
    # 验证输出文件是否存在和大小
    assert output_path.exists(), f"Output file {output_path} was not created"
    assert output_path.stat().st_size > 0, "Output file is empty"
    
    logger.info(f"Test completed successfully. Output saved to {output_path}")

if __name__ == "__main__":
    test_load_speaker_and_generate_audio() 