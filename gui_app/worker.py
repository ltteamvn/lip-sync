# -*- coding: utf-8 -*-
import os
import re
import sys
import glob
import time
import copy
import shutil
import pickle
import tempfile
import requests
import subprocess
import soundfile as sf
import numpy as np
import cv2
import torch
from argparse import Namespace
from tqdm import tqdm
from moviepy import VideoFileClip, AudioFileClip

from PyQt5.QtCore import QThread, pyqtSignal

# Monkeypatch basicsr / torchvision compatibility for GFPGAN
try:
    import torchvision
    import torchvision.transforms.functional as tv_F
    from types import ModuleType
    m = ModuleType("torchvision.transforms.functional_tensor")
    m.rgb_to_grayscale = tv_F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = m
except Exception as e:
    print(f"basicsr monkeypatch failed: {e}")

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', message='.*pretrained.*')
warnings.filterwarnings('ignore', message='.*weights.*deprecated.*')

_model_cache = {
    'vae': None,
    'unet': None,
    'pe': None,
    'whisper': None,
    'audio_processor': None,
    'gfpgan': None,
    'face_parsing': None,  # cache FaceParsing to avoid re-init overhead
    'face_parsing_params': None,  # track params to detect if re-init needed
}

class VideoWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.p = params
        self._stopped = False

    def stop(self):
        self._stopped = True

    def _log(self, msg):
        self.log.emit(msg)

    def _progress(self, val):
        self.progress.emit(val)

    def run(self):
        temp_dirs_to_clean = []
        try:
            self._log("═══════════════════════════════════")
            self._log("       NhepMieng - MUSETALK SOTA   ")
            self._log("═══════════════════════════════════")

            # ── Bước 1: Chuẩn bị tệp âm thanh ──────────────────────────────
            self._progress(5)
            driving_audio_path = ""
            if self.p['mode'] == 'script':
                self._log("📝 Đang chuyển đổi kịch bản sang giọng nói (edge-tts)...")
                tts_dir = tempfile.mkdtemp(prefix="musetalk_tts_")
                temp_dirs_to_clean.append(tts_dir)
                
                tts_mp3 = os.path.join(tts_dir, "tts_output.mp3")
                voice = self.p['voice']
                rate = self.p['rate']
                text = self.p['script']

                cmd_tts = ["edge-tts", "--voice", voice, "--text", text, "--write-media", tts_mp3]
                if rate != "+0%":
                    cmd_tts += ["--rate", rate]

                self._log(f"   > Chạy edge-tts: voice={voice}, rate={rate}")
                res = subprocess.run(cmd_tts, capture_output=True, text=True)
                if res.returncode != 0:
                    raise Exception(f"Không thể tạo âm thanh bằng edge-tts: {res.stderr}")

                # Convert MP3 to WAV 16kHz Mono (Whisper requirement)
                driving_audio_path = os.path.join(tts_dir, "tts_output.wav")
                ffmpeg_bin = self.p['ffmpeg_path']
                cmd_ffmpeg = [ffmpeg_bin, "-i", tts_mp3, "-ar", "16000", "-ac", "1", "-y", driving_audio_path]
                res_ff = subprocess.run(cmd_ffmpeg, capture_output=True)
                if res_ff.returncode != 0:
                    raise Exception("Không thể chuyển đổi âm thanh MP3 sang WAV 16kHz!")
                
                self._log("   ✅ TTS thành công.")
            else:
                raw_audio = self.p['audio_file']
                if not os.path.exists(raw_audio):
                    raise Exception(f"Không tìm thấy file âm thanh: {raw_audio}")
                
                tts_dir = tempfile.mkdtemp(prefix="musetalk_audio_")
                temp_dirs_to_clean.append(tts_dir)
                driving_audio_path = os.path.join(tts_dir, "audio.wav")
                
                ffmpeg_bin = self.p['ffmpeg_path']
                cmd_ffmpeg = [ffmpeg_bin, "-i", raw_audio, "-ar", "16000", "-ac", "1", "-y", driving_audio_path]
                res_ff = subprocess.run(cmd_ffmpeg, capture_output=True)
                if res_ff.returncode != 0:
                    raise Exception("Không thể chuyển đổi âm thanh sang WAV 16kHz!")
                self._log(f"✅ Sử dụng file âm thanh: {os.path.basename(raw_audio)}")

            if self._stopped:
                return

            # ── Bước 2: Xác định thời lượng và xử lý ảnh tĩnh ───────────────
            self._progress(15)
            audio_info = sf.info(driving_audio_path)
            duration = audio_info.duration
            self._log(f"⏱️ Thời lượng âm thanh: {duration:.2f}s")

            source_path = self.p['source_image']
            if not os.path.exists(source_path):
                raise Exception(f"Không tìm thấy tệp nguồn: {source_path}")

            # Kiểm tra xem là video hay ảnh
            from musetalk.utils.utils import get_file_type
            src_type = get_file_type(source_path)
            
            temp_frames_dir = None
            if src_type == "image":
                self._log("🖼️ Đầu vào là ảnh tĩnh. Đang nhân bản thành các khung hình...")
                temp_frames_dir = tempfile.mkdtemp(prefix="musetalk_frames_")
                temp_dirs_to_clean.append(temp_frames_dir)
                
                num_frames = max(1, int(duration * 25))
                self._log(f"   > Tạo ra {num_frames} khung hình cho video 25fps...")
                for i in range(num_frames):
                    dest_file = os.path.join(temp_frames_dir, f"{i:08d}.png")
                    shutil.copy(source_path, dest_file)
                video_input_path = temp_frames_dir
            elif src_type == "video":
                self._log("🎬 Đầu vào là video. Sẽ dùng trực tiếp các khung hình video gốc...")
                video_input_path = source_path
            else:
                raise Exception("Định dạng tệp nguồn không hỗ trợ (chỉ nhận ảnh hoặc video).")

            if self._stopped:
                return

            # ── Bước 3: Nạp mô hình MuseTalk (Sử dụng Cache) ─────────────────
            self._progress(25)
            self._log("🤖 Đang tải mô hình SOTA MuseTalk...")
            
            device = torch.device(self.p.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
            # Keep float32 for compatibility — unet and pe are not cast to float16,
            # mixing dtypes causes "Half and Float" mismatch during inference
            weight_dtype = torch.float32
            
            global _model_cache
            
            # Load MuseTalk load_all_model
            from musetalk.utils.utils import load_all_model
            from musetalk.utils.audio_processor import AudioProcessor
            from transformers import WhisperModel
            
            if _model_cache['vae'] is None:
                self._log("   > Tải VAE, U-Net, PositionalEncoding...")
                vae, unet, pe = load_all_model(
                    unet_model_path="./models/musetalk/pytorch_model.bin",
                    vae_type="sd-vae",
                    unet_config="./models/musetalk/musetalk.json",
                    device=device
                )
                _model_cache['vae'] = vae
                _model_cache['unet'] = unet
                _model_cache['pe'] = pe
            else:
                vae = _model_cache['vae']
                unet = _model_cache['unet']
                pe = _model_cache['pe']
                self._log("   > [Cache] Đã nạp VAE, U-Net.")

            if _model_cache['whisper'] is None:
                self._log("   > Tải Whisper Tiny Encoder...")
                whisper = WhisperModel.from_pretrained("./models/whisper")
                whisper = whisper.to(device=device, dtype=weight_dtype).eval()
                whisper.requires_grad_(False)
                _model_cache['whisper'] = whisper
            else:
                whisper = _model_cache['whisper']
                self._log("   > [Cache] Đã nạp Whisper Tiny.")

            if _model_cache['audio_processor'] is None:
                audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
                _model_cache['audio_processor'] = audio_processor
            else:
                audio_processor = _model_cache['audio_processor']

            # Tải GFPGAN nếu được yêu cầu
            use_enhancer = self.p.get('use_enhancer', True)
            if use_enhancer:
                if _model_cache['gfpgan'] is None:
                    # Tải trọng số GFPGANv1.4.pth nếu chưa tồn tại
                    gfpgan_weights_dir = "gfpgan/weights"
                    gfpgan_weights_path = os.path.join(gfpgan_weights_dir, "GFPGANv1.4.pth")
                    if not os.path.exists(gfpgan_weights_path):
                        self._log("📥 Không tìm thấy weights GFPGANv1.4.pth, đang tải tự động...")
                        os.makedirs(gfpgan_weights_dir, exist_ok=True)
                        gfpgan_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
                        response = requests.get(gfpgan_url, stream=True)
                        with open(gfpgan_weights_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        self._log("   ✅ Đã tải xong GFPGAN weights.")

                    self._log("   > Khởi tạo bộ khôi phục nét mặt GFPGAN...")
                    from gfpgan import GFPGANer
                    restorer = GFPGANer(
                        model_path=gfpgan_weights_path,
                        upscale=1,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=None
                    )
                    _model_cache['gfpgan'] = restorer
                else:
                    restorer = _model_cache['gfpgan']
                    self._log("   > [Cache] Đã nạp GFPGAN Restorer.")

            if self._stopped:
                return

            # ── Bước 4: Trích xuất đặc trưng âm thanh ───────────────────────
            self._progress(45)
            self._log("🎙️ Đang phân tích và trích xuất đặc trưng âm thanh...")
            whisper_input_features, librosa_length = audio_processor.get_audio_feature(driving_audio_path)
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features, 
                device, 
                weight_dtype, 
                whisper, 
                librosa_length,
                fps=25,
                audio_padding_length_left=2,
                audio_padding_length_right=2,
            )

            if self._stopped:
                return

            # ── Bước 5: Tiền xử lý hình ảnh ───────────────────────────────
            self._progress(60)
            self._log("👁️ Đang phát hiện vùng mặt và các điểm mốc (landmarks)...")
            
            # Đọc danh sách ảnh
            if src_type == "image":
                input_img_list = sorted(glob.glob(os.path.join(video_input_path, '*.[jpJP][pnPN]*[gG]')))
            else: # Video
                # Trích xuất khung hình từ video gốc sang thư mục tạm
                video_temp_dir = tempfile.mkdtemp(prefix="musetalk_vid_frames_")
                temp_dirs_to_clean.append(video_temp_dir)
                
                reader = imageio.get_reader(video_input_path)
                for idx, im in enumerate(reader):
                    imageio.imwrite(f"{video_temp_dir}/{idx:08d}.png", im)
                input_img_list = sorted(glob.glob(os.path.join(video_temp_dir, '*.[jpJP][pnPN]*[gG]')))
            
            from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
            
            bbox_shift = self.p.get('bbox_shift', 0)
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)

            # Lọc các khung hình không phát hiện được mặt
            valid_coords = []
            valid_frames = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox != coord_placeholder:
                    valid_coords.append(bbox)
                    valid_frames.append(frame)

            if len(valid_coords) == 0:
                raise Exception("Không phát hiện được bất kỳ khuôn mặt nào trong ảnh/video chân dung đầu vào!")

            # Chuẩn bị danh sách ảnh và khung mặt xoay vòng tương ứng với độ dài âm thanh
            input_latent_list = []
            self._log("🧬 Đang trích xuất đặc trưng VAE cho vùng mặt chân dung...")
            
            extra_margin = self.p.get('extra_margin', 10)
            with torch.no_grad():
                if src_type == "image" and len(valid_coords) > 0:
                    bbox = valid_coords[0]
                    frame = valid_frames[0]
                    x1, y1, x2, y2 = bbox
                    y2 = y2 + extra_margin
                    y2 = min(y2, frame.shape[0])
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    latents = vae.get_latents_for_unet(crop_frame)
                    input_latent_list = [latents] * len(valid_coords)
                else:
                    for bbox, frame in zip(valid_coords, valid_frames):
                        x1, y1, x2, y2 = bbox
                        y2 = y2 + extra_margin
                        y2 = min(y2, frame.shape[0])
                        crop_frame = frame[y1:y2, x1:x2]
                        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                        latents = vae.get_latents_for_unet(crop_frame)
                        input_latent_list.append(latents)

            # Xoay vòng khung hình để bù đủ độ dài âm thanh
            frame_list_cycle = valid_frames + valid_frames[::-1]
            coord_list_cycle = valid_coords + valid_coords[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            if self._stopped:
                return

            # ── Bước 6: Suy luận mô hình (Inference) ───────────────────────
            self._progress(75)
            self._log("🎬 Bắt đầu sinh chuyển động khẩu hình miệng bằng MuseTalk...")
            
            from musetalk.utils.utils import datagen
            video_num = len(whisper_chunks)
            batch_size = self.p.get('batch_size', 8)
            timesteps = torch.tensor([0], device=device)
            
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
                device=device,
            )
            
            res_frame_list = []
            num_batches = int(np.ceil(float(video_num) / batch_size))
            
            with torch.no_grad():
                for i, (whisper_batch, latent_batch) in enumerate(gen):
                    if self._stopped:
                        return
                    audio_feature_batch = pe(whisper_batch)
                    latent_batch = latent_batch.to(dtype=weight_dtype)
                    
                    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                    recon = vae.decode_latents(pred_latents)
                    for res_frame in recon:
                        res_frame_list.append(res_frame)
                    
                    # Cập nhật tiến trình chạy batch (75% -> 90%)
                    curr_progress = 75 + int((i / num_batches) * 15)
                    self._progress(curr_progress)

            # ── Bước 7: Khâu ghép mặt & Làm nét GFPGAN ──────────────────────
            self._progress(90)
            self._log("🧩 Đang ghép nối khẩu hình và phục hồi nét khuôn mặt...")
            
            from musetalk.utils.face_parsing import FaceParsing
            from musetalk.utils.blending import get_image

            # Cache FaceParsing — it loads a segmentation model internally,
            # re-initializing it every run is a significant overhead
            fp_params = (self.p.get('left_cheek_width', 90), self.p.get('right_cheek_width', 90))
            if _model_cache['face_parsing'] is None or _model_cache['face_parsing_params'] != fp_params:
                fp = FaceParsing(
                    left_cheek_width=fp_params[0],
                    right_cheek_width=fp_params[1]
                )
                _model_cache['face_parsing'] = fp
                _model_cache['face_parsing_params'] = fp_params
            else:
                fp = _model_cache['face_parsing']
            
            output_frames_dir = tempfile.mkdtemp(prefix="musetalk_out_")
            temp_dirs_to_clean.append(output_frames_dir)
            
            parsing_mode = self.p.get('parsing_mode', 'jaw')
            
            for idx, res_frame in enumerate(res_frame_list):
                if self._stopped:
                    return
                
                bbox = coord_list_cycle[idx % len(coord_list_cycle)]
                ori_frame = copy.deepcopy(frame_list_cycle[idx % len(frame_list_cycle)])
                x1, y1, x2, y2 = bbox
                y2 = y2 + extra_margin
                y2 = min(y2, ori_frame.shape[0])
                
                try:
                    res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except:
                    continue
                
                # Trộn vùng miệng tái tạo với khung mặt gốc
                combine_frame = get_image(ori_frame, res_frame_resized, [x1, y1, x2, y2], mode=parsing_mode, fp=fp)
                
                # Làm nét bằng GFPGAN nếu được yêu cầu
                if use_enhancer:
                    _, _, restored_img = restorer.enhance(combine_frame, has_aligned=False, only_center_face=False, paste_back=True)
                    combine_frame = restored_img
                
                cv2.imwrite(f"{output_frames_dir}/{idx:08d}.png", combine_frame)

            # ── Bước 8: Xuất video qua FFmpeg pipe trực tiếp ────────────────
            self._progress(95)
            self._log("🎞️ Ghép nối âm thanh và xuất video thành phẩm qua FFmpeg...")

            output_dir = self.p['output_dir']
            os.makedirs(output_dir, exist_ok=True)

            output_basename = f"nhepmieng_{int(time.time())}.mp4"
            final_output_path = os.path.abspath(os.path.join(output_dir, output_basename))

            files = sorted(glob.glob(f"{output_frames_dir}/[0-9]*.png"))
            if not files:
                raise Exception("Không có khung hình nào để xuất video!")

            # Read first frame to get dimensions
            sample = cv2.imread(files[0])
            h, w = sample.shape[:2]
            ffmpeg_bin = self.p['ffmpeg_path']

            # Single FFmpeg call: pipe raw BGR frames + mix audio → final mp4
            # Avoids double-encode overhead of imageio (write PNGs) + MoviePy (re-encode)
            cmd_out = [
                ffmpeg_bin,
                '-y',
                # Video input: raw BGR frames from stdin
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{w}x{h}',
                '-pix_fmt', 'bgr24',
                '-r', '25',
                '-i', 'pipe:0',
                # Audio input
                '-i', driving_audio_path,
                # Output settings
                '-vcodec', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-acodec', 'aac',
                '-b:a', '192k',
                '-shortest',
                final_output_path
            ]

            ffmpeg_proc = subprocess.Popen(cmd_out, stdin=subprocess.PIPE,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for f_path in files:
                if self._stopped:
                    ffmpeg_proc.stdin.close()
                    ffmpeg_proc.wait()
                    return
                frame_bgr = cv2.imread(f_path)
                if frame_bgr is not None:
                    ffmpeg_proc.stdin.write(frame_bgr.tobytes())

            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

            if ffmpeg_proc.returncode != 0:
                raise Exception("FFmpeg xuất video thất bại!")

            self._progress(100)
            self._log(f"🎉 Xuất video hoàn tất: {final_output_path}")
            self.finished.emit(final_output_path)

        except Exception as ex:
            import traceback
            traceback.print_exc()
            self.error.emit(str(ex))
        finally:
            # Dọn dẹp các thư mục tạm
            for t_dir in temp_dirs_to_clean:
                try:
                    if os.path.exists(t_dir):
                        shutil.rmtree(t_dir)
                except Exception as clean_err:
                    print(f"Lỗi khi dọn dẹp thư mục tạm {t_dir}: {clean_err}")
