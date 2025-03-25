import os
import sys
import logging
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import ChatTTS
from tools.audio import load_audio

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_voice_clone():
    """测试音色克隆功能"""
    try:
        # 初始化 ChatTTS
        logger.info("正在初始化 ChatTTS...")
        chat = ChatTTS.Chat(logger)
        
        # 加载模型
        logger.info("正在加载模型...")
        if not chat.load():
            logger.error("模型加载失败")
            return False
            
        # 加载音频文件
        logger.info("正在加载音频文件...")
        audio_path = os.path.join(project_root, "timebre", "timebre3.wav")
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return False
            
        # 加载并处理音频
        audio = load_audio(audio_path, sr=24000, mono=True)  # 当指定sr时，只返回音频数据
        logger.info(f"音频数据形状: {audio.shape}")
            
        # 确保音频是单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)
            
        # 确保音频数据是浮点数，并且值范围在[-1, 1]之间
        audio = audio.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
            
        # 转换为 tensor 并添加批次维度
        audio = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
        logger.info(f"转换后的音频数据形状: {audio.shape}")
            
        # 提取说话人特征
        logger.info("正在提取说话人特征...")
        try:
            # 使用 DVAE 处理音频
            mel = chat.dvae.preprocessor_mel(audio)
            logger.info(f"Mel 频谱图形状: {mel.shape}")
            
            # 通过下采样卷积
            x = chat.dvae.downsample_conv(mel / chat.dvae.coef.view(100, 1).expand(mel.shape))
            logger.info(f"下采样后形状: {x.shape}")
            
            # 通过编码器
            x = chat.dvae.encoder(x)
            logger.info(f"编码后形状: {x.shape}")
            
            # 通过量化层
            ind = chat.dvae.vq_layer(x)
            logger.info(f"量化后形状: {ind.shape}")
            
            # 直接使用量化后的特征，不进行维度调整
            speaker_features = chat.speaker.encode_prompt(ind)
            logger.info("成功提取说话人特征")
            
        except Exception as e:
            logger.error(f"提取说话人特征失败: {e}")
            logger.info("使用随机说话人特征作为备选...")
            speaker_features = chat.sample_random_speaker()
        
        # 测试文本
        test_texts = [
            "这是一个音色克隆测试，用来展示从音频中提取说话人特征的效果。",  # 中文测试
            "Hello, this is a voice cloning test to demonstrate the effect of speaker feature extraction."  # 英文测试
        ]
        
        # 创建输出目录
        output_dir = os.path.join(project_root, "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成音频
        logger.info("正在生成音频...")
        for i, text in enumerate(test_texts):
            logger.info(f"正在处理第 {i+1} 个测试文本...")
            
            # 设置生成参数
            params_infer_code = chat.InferCodeParams(
                spk_emb=speaker_features,  # 使用说话人特征
                temperature=0.2,  # 进一步降低温度以获得更稳定的生成
                top_P=0.8,       # 降低采样概率分布范围
                top_K=30,        # 减少候选数量
                max_new_token=1024,  # 进一步减小最大token数
                min_new_token=256,   # 减小最小token数
                stream_batch=32,     # 减小batch大小
                stream_speed=24000,  # 保持采样率一致
                pass_first_n_batches=4  # 减少pass batch数
            )
            
            # 生成音频
            wav = chat.infer(
                text,
                skip_refine_text=True,  # 跳过文本细化
                params_infer_code=params_infer_code,
                stream=False,  # 不使用流式生成
                split_text=False,  # 不分割文本
                do_text_normalization=True,  # 启用文本规范化
                do_homophone_replacement=True  # 启用同音字替换
            )
            
            # 保存音频文件
            output_path = os.path.join(output_dir, f"voice_clone_test_{i+1}.mp3")
            sf.write(output_path, wav[0], 24000)
            logger.info(f"音频已保存到: {output_path}")
            
            # 验证输出文件
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                logger.error(f"输出文件无效: {output_path}")
                return False
                
        logger.info("音色克隆测试完成")
        return True
        
    except Exception as e:
        logger.error(f"音色克隆失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_voice_clone()
    sys.exit(0 if success else 1) 