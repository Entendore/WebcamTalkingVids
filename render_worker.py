import os
import cv2
import numpy as np
import scipy.io.wavfile as wav
from moviepy import VideoFileClip, AudioFileClip
from PySide6.QtCore import QThread, Signal
from audio_engine import AudioEngine
from video_engine import VideoEngine
from datetime import datetime
import logging

class RenderWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.audio_engine = AudioEngine()
        self.video_engine = VideoEngine()

    def run(self):
        try:
            self.progress.emit(10)
            logging.info("RenderWorker: Starting audio processing...")

            # Audio Processing
            if len(self.params['audio_data']) > 0:
                audio_np = np.concatenate(self.params['audio_data'], axis=0).flatten().astype(np.float32)
                logging.info(f"RenderWorker - Captured Audio Samples: {len(audio_np)}")
                logging.info(f"RenderWorker - Raw Audio Max Amplitude: {np.max(np.abs(audio_np)) if len(audio_np) > 0 else 0}")
                
                # Guard against completely silent raw audio (which can break processing chains)
                if len(audio_np) > 0 and np.max(np.abs(audio_np)) > 1e-6:
                    processed_audio = self.audio_engine.process_audio_chain(
                        audio_np, 
                        self.params['samplerate'], 
                        self.params['audio_settings']
                    )
                else:
                    logging.warning("RenderWorker - Raw audio is silent. Skipping processing.")
                    processed_audio = audio_np
                
                # Guard against processed audio resulting in empty array
                if len(processed_audio) == 0:
                    logging.warning("RenderWorker - Processed audio is empty. Creating 1s silent fallback.")
                    processed_audio = np.zeros(self.params['samplerate'], dtype=np.float32)

                logging.info(f"RenderWorker - Processed Audio Max Amplitude: {np.max(np.abs(processed_audio))}")
                wav.write(self.params['processed_audio_path'], self.params['samplerate'], (processed_audio * 32767).astype(np.int16))
            else:
                logging.warning("RenderWorker - No audio recorded. Creating 1s silent fallback.")
                dummy_audio = np.zeros(self.params['samplerate'], dtype=np.float32)
                wav.write(self.params['processed_audio_path'], self.params['samplerate'], (dummy_audio * 32767).astype(np.int16))

            self.progress.emit(20)
            logging.info("RenderWorker: Starting video processing...")

            # Video Processing
            cap = cv2.VideoCapture(self.params['raw_video_path'])
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(self.params['processed_video_path'], cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            self.video_engine.edge_accumulator = None
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                proc_frame = self.video_engine.apply_video_adjustments(frame, self.params['video_settings'])
                out.write(proc_frame)
                current_frame += 1
                if total_frames > 0:
                    self.progress.emit(20 + int((current_frame / total_frames) * 60))
            
            cap.release()
            out.release()
            logging.info(f"RenderWorker: Video processing complete. Processed {current_frame} frames.")

            self.progress.emit(85)

            # Merge Audio + Video & Export
            logging.info("RenderWorker: Merging audio and video for final export...")
            video_clip = VideoFileClip(self.params['processed_video_path'])
            audio_clip = AudioFileClip(self.params['processed_audio_path'])
            final_clip = video_clip.with_audio(audio_clip)
            final_clip.write_videofile(
                self.params['save_path'], 
                codec=self.params['ffmpeg_codec'], 
                audio_codec="aac", 
                temp_audiofile="temp_audio.m4a", 
                remove_temp=True, 
                logger=None,
                ffmpeg_params=['-preset', self.params['render_preset']]
            )
            
            # Markers
            if self.params.get('markers'):
                marker_path = self.params['save_path'].replace('.mp4', '_markers.txt')
                with open(marker_path, 'w') as f:
                    f.write("--- Highlight Markers ---\n")
                    for m_time in self.params['markers']:
                        m, s = divmod(int(m_time), 60)
                        h, m = divmod(m, 60)
                        f.write(f"[{h:02d}:{m:02d}:{s:02d}] Highlight\n")
            
            self.progress.emit(100)
            logging.info(f"RenderWorker: Successfully exported video to {self.params['save_path']}")
            self.finished.emit(self.params['save_path'])

        except Exception as e:
            logging.error(f"RenderWorker: Processing failed - {e}", exc_info=True)
            self.error.emit(str(e))
        finally:
            for f in [self.params['raw_video_path'], self.params.get('raw_audio_path'), self.params['processed_video_path'], self.params['processed_audio_path']]:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass