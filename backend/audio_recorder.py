#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频录制工具
支持列出所有音频输入设备，选择设备并录制指定时长的WAV文件
"""

import pyaudio
import wave
import os
from datetime import datetime


def list_audio_input_devices(verbose=True):
    """列出所有可用的音频输入设备"""
    audio = pyaudio.PyAudio()
    devices = []
    
    if verbose:
        print("\n=== 可用的音频输入设备 ===\n")
    
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        # 检查设备是否有输入通道
        if device_info['maxInputChannels'] > 0:
            devices.append({
                'index': i,
                'name': device_info['name'],
                'channels': device_info['maxInputChannels'],
                'sample_rate': int(device_info['defaultSampleRate'])
            })
            if verbose:
                print(f"设备 {i}: {device_info['name']}")
                print(f"  通道数: {device_info['maxInputChannels']}")
                print(f"  默认采样率: {int(device_info['defaultSampleRate'])} Hz")
                print()
    
    audio.terminate()
    return devices


def select_input_device(devices):
    """让用户选择输入设备"""
    if not devices:
        print("错误: 未找到可用的音频输入设备！")
        return None
    
    while True:
        try:
            choice = input(f"请选择设备编号 (0-{len(devices)-1}): ")
            device_index = int(choice)
            
            if 0 <= device_index < len(devices):
                selected_device = devices[device_index]
                print(f"\n已选择设备: {selected_device['name']}")
                print(f"设备索引: {selected_device['index']}")
                return selected_device
            else:
                print(f"无效的选择，请输入 0 到 {len(devices)-1} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n操作已取消")
            return None


def get_recording_duration():
    """获取用户输入的录制时长（秒）"""
    while True:
        try:
            duration = float(input("\n请输入录制时长（秒）: "))
            if duration > 0:
                return duration
            else:
                print("录制时长必须大于0")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n操作已取消")
            return None


def get_save_path():
    """获取用户输入的保存路径"""
    default_path = os.path.join(os.getcwd(), "recordings")
    
    print(f"\n默认保存目录: {default_path}")
    custom_path = input("输入自定义保存路径（直接回车使用默认）: ").strip()
    
    if not custom_path:
        save_path = default_path
    else:
        save_path = custom_path
    
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    
    return save_path


def record_audio(device_index, duration, sample_rate=44100, channels=1, chunk=1024, verbose=True):
    """
    录制音频
    
    参数:
        device_index: 输入设备索引
        duration: 录制时长（秒）
        sample_rate: 采样率（默认44100 Hz）
        channels: 声道数（默认1，单声道）
        chunk: 每次读取的帧数（默认1024）
        verbose: 是否打印进度信息（默认True）
    """
    audio = pyaudio.PyAudio()
    
    # 打开音频流
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk
    )
    
    if verbose:
        print(f"\n开始录制... (时长: {duration} 秒)")
        print("录制中... (按 Ctrl+C 可提前停止)")
    
    frames = []
    
    try:
        # 计算需要读取的帧数
        total_frames = int(sample_rate / chunk * duration)
        
        for i in range(total_frames):
            data = stream.read(chunk)
            frames.append(data)
            # 显示进度
            if verbose:
                progress = (i + 1) / total_frames * 100
                print(f"\r进度: {progress:.1f}%", end='', flush=True)
        
        if verbose:
            print("\n录制完成！")
        
    except KeyboardInterrupt:
        if verbose:
            print("\n\n录制已提前停止")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    return frames


def save_wav_file(frames, save_path, sample_rate=44100, channels=1, verbose=True):
    """保存音频为WAV文件"""
    # 生成文件名（使用时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    filepath = os.path.join(save_path, filename)
    
    # 保存WAV文件
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)  # 16位 = 2字节
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    if verbose:
        print(f"\n音频已保存到: {filepath}")
        print(f"文件大小: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    return filepath


def main():
    """主函数"""
    print("=" * 50)
    print("音频录制工具")
    print("=" * 50)
    
    try:
        # 1. 列出所有输入设备
        devices = list_audio_input_devices()
        
        if not devices:
            print("错误: 未找到可用的音频输入设备！")
            return
        
        # 2. 选择设备
        selected_device = select_input_device(devices)
        if not selected_device:
            return
        
        # 3. 获取录制时长
        duration = get_recording_duration()
        if duration is None:
            return
        
        # 4. 获取保存路径
        save_path = get_save_path()
        
        # 5. 录制音频
        frames = record_audio(
            device_index=selected_device['index'],
            duration=duration,
            sample_rate=selected_device['sample_rate'],
            channels=min(selected_device['channels'], 2)  # 最多使用2个通道（立体声）
        )
        
        if not frames:
            print("未录制到音频数据")
            return
        
        # 6. 保存WAV文件
        save_wav_file(
            frames=frames,
            save_path=save_path,
            sample_rate=selected_device['sample_rate'],
            channels=min(selected_device['channels'], 2)
        )
        
        print("\n操作完成！")
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

