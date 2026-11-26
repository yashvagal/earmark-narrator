// Makes a stand-in reference clip with a built-in Kokoro voice, so the test never uses anyone's real voice.
import { KokoroTTS } from 'kokoro-js';
import { writeFileSync } from 'node:fs';
const text = 'The river ran quietly past the old mill. By morning the fog had lifted, and the whole valley was bright and still. We walked along the bank and talked about the long winter ahead.';
const tts = await KokoroTTS.from_pretrained('onnx-community/Kokoro-82M-v1.0-ONNX', { dtype: 'fp32', device: 'cpu' });
const audio = await tts.generate(text, { voice: 'am_michael' });
audio.save('speedtest/ref.wav');
writeFileSync('speedtest/ref.txt', text);
console.log(`reference clip: ${(audio.audio.length / audio.sampling_rate).toFixed(1)} s`);
