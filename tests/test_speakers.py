import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import torch
import torchaudio
import ChatTTS
from tools.logger import get_logger
from tools.audio import pcm_arr_to_mp3_view

logger = get_logger("SpeakerTest")

def test_speakers():
    # 初始化ChatTTS
    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("正在初始化ChatTTS...")
    
    # 加载模型
    if chat.load(source="local"):
        logger.info("模型加载成功")
    else:
        logger.error("模型加载失败")
        return
    
    # 生成多个随机音色进行测试
    speakers = [chat.sample_random_speaker() for _ in range(3)]
    logger.info(f"生成了 {len(speakers)} 个随机音色样本")
    
    # 打印所有音色信息
    for i, speaker in enumerate(speakers, 1):
        logger.info(f"音色 {i}: {speaker}")
    
    # 选择测试文本
    test_texts = [
        "这是一个测试语音，用来展示不同的音色效果。",
        "Hello, this is a test voice to demonstrate different speaker effects."
    ]
    
    # 为每个音色生成测试音频
    for i, speaker in enumerate(speakers, 1):
        logger.info(f"正在使用音色 {speaker} 生成测试音频...")
        
        # 生成音频
        wavs = chat.infer(
            test_texts,
            params_infer_code=ChatTTS.Chat.InferCodeParams(
                spk_emb=speaker,
            ),
        )
        
        # 保存音频文件
        for j, wav in enumerate(wavs):
            # 使用绝对路径保存音频文件
            output_dir = os.path.join(os.getcwd(), "test_outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, f"speaker_{i}_sample_{j+1}.mp3")
            try:
                data = pcm_arr_to_mp3_view(wav)
                with open(output_filename, "wb") as f:
                    f.write(data)
                logger.info(f"已保存音频文件: {output_filename}")
            except Exception as e:
                logger.error(f"保存音频文件失败: {e}")

if __name__ == "__main__":
    test_speakers()
    logger.info("音色测试完成")