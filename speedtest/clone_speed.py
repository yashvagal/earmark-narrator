# Times Qwen3-TTS 0.6B voice cloning on this runner's CPU. Prints only timings, never text or audio.
import os, sys, time
import soundfile as sf
import torch

threads = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[2] if len(sys.argv) > 2 else "fp32"]
torch.set_num_threads(threads)
print(f"cpus {os.cpu_count()}, threads {threads}, dtype {dtype}")

t = time.time()
from qwen_tts import Qwen3TTSModel
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cpu", dtype=dtype, attn_implementation="sdpa")
print(f"model load {time.time() - t:.1f} s")

audio, sr = sf.read("speedtest/ref.wav", dtype="float32", always_2d=False)
t = time.time()
prompt = model.create_voice_clone_prompt(ref_audio=(audio, sr), ref_text=open("speedtest/ref.txt").read().strip(), x_vector_only_mode=False)
print(f"voice prompt {time.time() - t:.1f} s")

# About 260 characters each, the size Earmark's narrator reads at a time.
pieces = [
    "The house at the end of the lane had been empty for years, but every spring someone still came to cut the grass. Nobody in the village knew who it was, and after a while nobody thought to ask. It was simply one of the things that happened, like the swallows returning.",
    "She kept a small notebook in her coat pocket and wrote down anything that surprised her: a word she had never heard, the price of bread in a town she was passing through, the exact color of the sea on a cold morning. By the end of the year it was nearly full.",
    "There are two ways to learn a city. You can study the map until the streets line up in your head, or you can get lost on purpose and let the corners teach you. He had always preferred the second way, even when it made him late for nearly everything that mattered.",
    "The meeting went on longer than anyone expected. By the time the last question was answered, the coffee had gone cold and the light outside had turned orange. Still, nobody seemed to mind very much. They had finally agreed on something, and that felt like enough for one day.",
    "When the storm finally passed, the fields were flattened and the road was covered in branches. Neighbors came out with saws and wheelbarrows without being asked. By noon the road was clear again, and someone had set up a table with sandwiches and a big pot of soup.",
]
total_audio = total_time = 0.0
for i, text in enumerate(pieces, 1):
    t = time.time()
    wavs, out_sr = model.generate_voice_clone(text=text, language="English", voice_clone_prompt=prompt)
    took = time.time() - t
    dur = len(wavs[0]) / out_sr
    total_audio += dur
    total_time += took
    print(f"piece {i}: {dur:.1f} s of audio in {took:.1f} s ({dur / took:.2f}x real time)")
print(f"TOTAL: {total_audio:.1f} s of audio in {total_time:.1f} s = {total_audio / total_time:.2f}x real time")
