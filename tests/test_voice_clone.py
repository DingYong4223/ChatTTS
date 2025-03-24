import os
import sys
import logging
import torch
import torchaudio
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from ChatTTS import Chat
from tools.logger import get_logger
from tools.audio import pcm_arr_to_mp3_view

def test_voice_clone():
    # 设置日志
    logger = get_logger("test_voice_clone")
    
    # 初始化ChatTTS
    chat = Chat(logger)
    
    # 加载模型
    chat.load(source="local")
    
    # 创建输出目录
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 测试文本
    test_texts = [
        "这是一个音色克隆测试，用来展示从音频中提取说话人特征的效果。",
        "Hello, this is a voice cloning test to demonstrate the effect of speaker feature extraction."
    ]
    
    # 从音频文件中提取说话人特征
    source_audio_path = "timebre/timebre3.wav"  # 使用项目中的示例音频
    logger.info(f"正在从音频文件 {source_audio_path} 中提取说话人特征...")
    
    try:
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(source_audio_path)
        if sample_rate != 24000:
            resampler = torchaudio.transforms.Resample(sample_rate, 24000)
            waveform = resampler(waveform)
        
        # 确保音频是单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 提取说话人特征
        try:
            # 将音频数据转换为正确的维度
            waveform = waveform.squeeze(0)  # 移除批次维度
            speaker_embedding = chat.sample_audio_speaker(waveform.numpy())
            logging.info("成功提取说话人特征")
            
            # 生成测试文本的语音
            for i, text in enumerate(test_texts):
                logging.info(f"正在生成文本: {text}")
                wavs = chat.infer(
                    text=text,
                    params_infer_code=chat.InferCodeParams(spk_smp=speaker_embedding)
                )
                
                # 保存音频文件
                output_path = os.path.join(output_dir, f"voice_clone_test_{i+1}.mp3")
                data = pcm_arr_to_mp3_view(wavs[0])  # 取第一个音频
                with open(output_path, "wb") as f:
                    f.write(data)
                logging.info(f"已保存音频文件: {output_path}")
                
                # 验证输出文件是否存在
                assert os.path.exists(output_path), f"Output file {output_path} was not created"
                assert os.path.getsize(output_path) > 0, f"Output file {output_path} is empty"
        except Exception as e:
            logger.error(f"提取说话人特征失败: {e}")
            raise
        
        # 验证所有输出文件是否存在
        for i in range(len(test_texts)):
            output_path = os.path.join(output_dir, f"voice_clone_test_{i+1}.mp3")
            assert os.path.exists(output_path), f"Output file {output_path} was not created"
            assert os.path.getsize(output_path) > 0, f"Output file {output_path} is empty"
        
        logger.info(f"Test completed successfully. Output saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"音色克隆失败: {e}")
        raise

if __name__ == "__main__":
    test_voice_clone() 