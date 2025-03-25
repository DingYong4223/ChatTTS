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
    
    # 测试文本 - 分别测试中文和英文
    test_text = "你好，这是一个测试音频。"
    test_text_en = "Hello, this is a test audio."
    
    # 从文件加载音色
    speaker_save_path = output_dir / "speaker_embedding.txt"
    loaded_speaker_embedding = load_speaker_embedding(speaker_save_path, logger)
    
    # 设置生成参数
    params = chat.InferCodeParams(
        spk_emb=loaded_speaker_embedding,
        manual_seed=42,  # 固定随机种子
        temperature=0.3,  # 提高温度参数，增加生成多样性
        top_P=0.95,  # 提高top_P值，使采样更集中
        top_K=50,  # 增加top_K值，扩大采样范围
        repetition_penalty=1.1,  # 略微提高重复惩罚
        prompt="[speed_5]"  # 添加速度提示
    )
    
    # 使用加载的音色生成中文语音
    output_path_cn = output_dir / "speaker_load_test_cn.mp3"
    wavs_cn = chat.infer(
        text=test_text,
        params_infer_code=params,
        split_text=False,  # 禁用文本分割
        do_text_normalization=True,  # 保持文本标准化
        do_homophone_replacement=True  # 保持同音字替换
    )
    
    # 保存中文音频文件
    if isinstance(wavs_cn, list):
        wavs_cn = wavs_cn[0]  # 如果返回多个音频，取第一个
    try:
        data_cn = pcm_arr_to_mp3_view(wavs_cn)
        with open(output_path_cn, "wb") as f:
            f.write(data_cn)
        logger.info(f"已保存中文音频文件: {output_path_cn}")
    except Exception as e:
        logger.error(f"保存中文音频文件失败: {e}")
        raise
    
    # 使用加载的音色生成英文语音
    output_path_en = output_dir / "speaker_load_test_en.mp3"
    wavs_en = chat.infer(
        text=test_text_en,
        params_infer_code=params,
        split_text=False,  # 禁用文本分割
        do_text_normalization=True,  # 保持文本标准化
        do_homophone_replacement=True  # 保持同音字替换
    )
    
    # 保存英文音频文件
    if isinstance(wavs_en, list):
        wavs_en = wavs_en[0]  # 如果返回多个音频，取第一个
    try:
        data_en = pcm_arr_to_mp3_view(wavs_en)
        with open(output_path_en, "wb") as f:
            f.write(data_en)
        logger.info(f"已保存英文音频文件: {output_path_en}")
    except Exception as e:
        logger.error(f"保存英文音频文件失败: {e}")
        raise
    
    # 验证输出文件是否存在和大小
    assert output_path_cn.exists(), f"Output file {output_path_cn} was not created"
    assert output_path_en.exists(), f"Output file {output_path_en} was not created"
    assert output_path_cn.stat().st_size > 0, "Chinese output file is empty"
    assert output_path_en.stat().st_size > 0, "English output file is empty"
    
    logger.info(f"Test completed successfully. Outputs saved to {output_path_cn} and {output_path_en}")

if __name__ == "__main__":
    test_load_speaker_and_generate_audio() 