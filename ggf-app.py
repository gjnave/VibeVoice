import os
import time
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import traceback
import threading
import requests
import json
from datetime import datetime
from pathlib import Path

from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from processor.vibevoice_processor import VibeVoiceProcessor
from modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class ConfigManager:
    def __init__(self, config_file="vibevoice_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "openrouter_api_key": "",
            "llm_provider": "ollama",
            "ollama_model": "llama2",
            "openrouter_model": "anthropic/claude-3.5-sonnet"
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    default_config.update(saved_config)
                print(f"✅ Loaded config from {self.config_file}")
            except Exception as e:
                print(f"❌ Error loading config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Saved config to {self.config_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            return False
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value
        return self.save_config()

class VibeVoiceDemo:
    def __init__(self, model_paths: dict, device: str = "cuda", inference_steps: int = 5):
        """
        model_paths: dict like {"VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
                                "VibeVoice-1.1B": "microsoft/VibeVoice-1.1B"}
        """
        self.model_paths = model_paths
        self.device = device
        self.inference_steps = inference_steps
        self.config_manager = ConfigManager()

        self.is_generating = False

        # Multi-model holders
        self.models = {}        # name -> model
        self.processors = {}    # name -> processor
        self.current_model_name = None

        self.available_voices = {}

        self.load_models()          # load all on CPU
        self.setup_voice_presets()
        self.load_example_scripts()

    def load_models(self):
        print("Loading processors and models on CPU...")
        for name, path in self.model_paths.items():
            print(f" - {name} from {path}")
            proc = VibeVoiceProcessor.from_pretrained(path)
            mdl = VibeVoiceForConditionalGenerationInference.from_pretrained(
                path, torch_dtype=torch.bfloat16
            )
            # Keep on CPU initially
            self.processors[name] = proc
            self.models[name] = mdl
        # choose default
        self.current_model_name = next(iter(self.models))
        print(f"Default model is {self.current_model_name}")

    def _place_model(self, target_name: str):
        """
        Move the selected model to CUDA and push all others back to CPU.
        """
        for name, mdl in self.models.items():
            if name == target_name:
                self.models[name] = mdl.to(self.device)
            else:
                self.models[name] = mdl.to("cpu")
        self.current_model_name = target_name
        print(f"Model {target_name} is now on {self.device}. Others moved to CPU.")

    def setup_voice_presets(self):
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            return
        wav_files = [f for f in os.listdir(voices_dir)
                     if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'))]
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            self.available_voices[name] = os.path.join(voices_dir, wav_file)
        print(f"Voices loaded: {list(self.available_voices.keys())}")

    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])

    def call_llm_api(self, topic: str, num_speakers: int, provider: str, api_key: str = None, model: str = None) -> str:
        """
        Call LLM API (OpenRouter or Ollama) to generate podcast script
        """
        try:
            if provider == "openrouter":
                return self._call_openrouter(topic, num_speakers, api_key, model)
            elif provider == "ollama":
                return self._call_ollama(topic, num_speakers, model)
            else:
                return f"Error: Unknown provider {provider}"
                
        except Exception as e:
            return f"Error calling LLM API: {str(e)}"

    def _call_openrouter(self, topic: str, num_speakers: int, api_key: str, model: str) -> str:
        """Call OpenRouter API"""
        if not api_key:
            return "Error: OpenRouter API key is required. Please enter your API key in the settings."
        
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        prompt = self._build_prompt(topic, num_speakers)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/vibevoice/vibevoice",
            "X-Title": "VibeVoice Podcast Generator"
        }
            
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2000
        }
        
        response = requests.post(api_url, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: API request failed with status {response.status_code}: {response.text}"

    def _call_ollama(self, topic: str, num_speakers: int, model: str) -> str:
        """Call Ollama API"""
        api_url = "http://localhost:11434/api/generate"
        
        prompt = self._build_prompt(topic, num_speakers)
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(api_url, json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return f"Error: Ollama request failed with status {response.status_code}. Make sure Ollama is running on localhost:11434"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Please make sure Ollama is installed and running on localhost:11434"

    def _build_prompt(self, topic: str, num_speakers: int) -> str:
        """Build the prompt for LLM"""
        base_prompt = f"""Create a engaging podcast transcript about: {topic}

Format requirements:
- Use exactly {num_speakers} speakers
- Format as: Speaker 1: ... Speaker 2: ... 
- Make it conversational and natural
- Include 5-10 exchanges between speakers
- Cover different aspects of the topic
- End with a concluding thought

Podcast transcript:"""
        return base_prompt

    @torch.inference_mode()
    def generate_podcast(self,
                         num_speakers: int,
                         script: str,
                         speaker_1: str = None,
                         speaker_2: str = None,
                         speaker_3: str = None,
                         speaker_4: str = None,
                         cfg_scale: float = 1.3,
                         model_name: str = None):
        """
        Generates a podcast as a single audio file from a script and saves it.
        Non-streaming.
        """
        try:
            # pick model
            model_name = model_name or self.current_model_name
            if model_name not in self.models:
                raise gr.Error(f"Unknown model: {model_name}")

            # place models on devices
            self._place_model(model_name)
            model = self.models[model_name]
            processor = self.processors[model_name]

            print(f"Using model {model_name} on {self.device}")

            model.eval()
            model.set_ddpm_inference_steps(num_steps=self.inference_steps)

            self.is_generating = True

            if not script.strip():
                raise gr.Error("Error: Please provide a script.")

            script = script.replace("'", "'")

            if not 1 <= num_speakers <= 4:
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")

            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            for i, speaker_name in enumerate(selected_speakers):
                if not speaker_name or speaker_name not in self.available_voices:
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")

            log = f"🎙️ Generating podcast with {num_speakers} speakers\n"
            log += f"🧠 Model: {model_name}\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}\n"
            log += f"🎭 Speakers: {', '.join(selected_speakers)}\n"

            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    raise gr.Error(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)

            log += f"✅ Loaded {len(voice_samples)} voice samples\n"

            lines = script.strip().split('\n')
            formatted_script_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")

            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 Formatted script with {len(formatted_script_lines)} turns\n"
            log += "🔄 Processing with VibeVoice...\n"

            inputs = processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False,
            )
            generation_time = time.time() - start_time

            if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs[0] is not None:
                audio_tensor = outputs.speech_outputs[0]
                audio = audio_tensor.cpu().float().numpy()
            else:
                raise gr.Error("❌ Error: No audio was generated by the model. Please try again.")

            if audio.ndim > 1:
                audio = audio.squeeze()

            sample_rate = 24000

            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(output_dir, f"podcast_{timestamp}.wav")
            sf.write(file_path, audio, sample_rate)
            print(f"💾 Podcast saved to {file_path}")

            total_duration = len(audio) / sample_rate
            log += f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
            log += f"🎵 Final audio duration: {total_duration:.2f} seconds\n"
            log += f"✅ Successfully saved podcast to: {file_path}\n"

            self.is_generating = False
            return (sample_rate, audio), log

        except gr.Error as e:
            self.is_generating = False
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            return None, error_msg

        except Exception as e:
            self.is_generating = False
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return None, error_msg


    @staticmethod
    def _infer_num_speakers_from_script(script: str) -> int:
        """
        Infer number of speakers by counting distinct 'Speaker X:' tags in the script.
        Robust to 0- or 1-indexed labels and repeated turns.
        Falls back to 1 if none found.
        """
        import re
        ids = re.findall(r'(?mi)^\s*Speaker\s+(\d+)\s*:', script)
        return len({int(x) for x in ids}) if ids else 1

    def load_example_scripts(self):
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        if not os.path.exists(examples_dir):
            return

        txt_files = sorted(
            [f for f in os.listdir(examples_dir) if f.lower().endswith('.txt')]
        )
        for txt_file in txt_files:
            try:
                with open(os.path.join(examples_dir, txt_file), 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                if script_content:
                    num_speakers = self._infer_num_speakers_from_script(script_content)
                    self.example_scripts.append([num_speakers, script_content])
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")


def convert_to_16_bit_wav(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.array(data)
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    return (data * 32767).astype(np.int16)


def create_demo_interface(demo_instance: VibeVoiceDemo):
    custom_css = """
    .llm-settings {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: #f9f9f9;
    }
    .api-key-input {
        font-family: monospace;
    }
    .success-message {
        color: #22c55e;
        font-weight: bold;
    }
    .error-message {
        color: #ef4444;
        font-weight: bold;
    }
    """

    with gr.Blocks(
        title="VibeVoice - AI Podcast Generator",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        )
    ) as interface:

        # Store API key in session state
        openrouter_api_key = gr.State(value=demo_instance.config_manager.get("openrouter_api_key", ""))
        llm_provider = gr.State(value=demo_instance.config_manager.get("llm_provider", "ollama"))
        ollama_model = gr.State(value=demo_instance.config_manager.get("ollama_model", "llama2"))
        openrouter_model = gr.State(value=demo_instance.config_manager.get("openrouter_model", "anthropic/claude-3.5-sonnet"))

        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Vibe Podcasting</h1>
            <p><a href="https://getgoingfast.pro" target="_blank">Get Going Fast</a></p>
            <p><a href="https://music.youtube.com/channel/UCGV4scbVcBqo2aVTy23JJeA" target="_blank">Listen to Good music</a></p>
            <p>Generating Long-form Multi-speaker AI Podcast with VibeVoice</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1, elem_classes="settings-card"):
                gr.Markdown("### 🎛️ Podcast Settings")

                model_dropdown = gr.Dropdown(
                    choices=list(demo_instance.models.keys()),
                    value=demo_instance.current_model_name,
                    label="Model",
                )

                num_speakers = gr.Slider(
                    minimum=1, maximum=4, value=2, step=1,
                    label="Number of Speakers",
                    elem_classes="slider-container"
                )

                gr.Markdown("### 🎭 Speaker Selection")
                available_speaker_names = list(demo_instance.available_voices.keys())
                default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']

                speaker_selections = []
                for i in range(4):
                    default_value = default_speakers[i] if i < len(default_speakers) else None
                    speaker = gr.Dropdown(
                        choices=available_speaker_names,
                        value=default_value,
                        label=f"Speaker {i+1}",
                        visible=(i < 2),
                        elem_classes="speaker-item"
                    )
                    speaker_selections.append(speaker)

                gr.Markdown("### ⚙️ Advanced Settings")
                with gr.Accordion("Generation Parameters", open=False):
                    cfg_scale = gr.Slider(
                        minimum=1.0, maximum=2.0, value=1.3, step=0.05,
                        label="CFG Scale (Guidance Strength)",
                        elem_classes="slider-container"
                    )

            with gr.Column(scale=2, elem_classes="generation-card"):
                gr.Markdown("### 📝 Script Input")
                
                # LLM Script Generation Section
                with gr.Accordion("🤖 Generate Script with LLM", open=False) as llm_accordion:
                    use_llm = gr.Checkbox(
                        label="Use LLM to generate script from topic",
                        value=False,
                        info="Generate podcast script automatically using AI"
                    )
                    
                    with gr.Group(visible=False) as llm_settings:
                        with gr.Row():
                            llm_provider_dropdown = gr.Dropdown(
                                choices=["ollama", "openrouter"],
                                value=llm_provider.value,
                                label="LLM Provider",
                                info="Choose your preferred LLM service"
                            )
                            
                            ollama_model_dropdown = gr.Dropdown(
                                choices=["llama2", "llama3", "mistral", "codellama", "phi3"],
                                value=ollama_model.value,
                                label="Ollama Model",
                                visible=(llm_provider.value == "ollama")
                            )
                            
                            openrouter_model_dropdown = gr.Dropdown(
                                choices=[
                                    "x-ai/grok-4-fast:free",
                                    "deepseek/deepseek-chat-v3.1:free", 
                                    "moonshotai/kimi-k2:free",
                                    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
                                    "meta-llama/llama-3.3-8b-instruct:free"
                                ],
                                value=openrouter_model.value,
                                label="OpenRouter Model",
                                visible=(llm_provider.value == "openrouter")
                            )
                        
                        with gr.Row(visible=(llm_provider.value == "openrouter")) as api_key_row:
                            api_key_input = gr.Textbox(
                                label="OpenRouter API Key",
                                value=openrouter_api_key.value,
                                type="password",
                                placeholder="Enter your OpenRouter API key...",
                                elem_classes="api-key-input"
                            )
                            save_key_btn = gr.Button("💾 Save Key", size="sm")
                            save_status = gr.HTML("")
                        
                        topic_input = gr.Textbox(
                            label="Podcast Topic",
                            placeholder="Enter a topic for your podcast (e.g., 'The future of artificial intelligence in healthcare')...",
                            lines=3,
                            elem_classes="topic-input"
                        )
                        
                        with gr.Row():
                            generate_script_btn = gr.Button(
                                "🚀 Generate Script with LLM",
                                variant="primary",
                                elem_classes="llm-btn"
                            )
                            clear_topic_btn = gr.Button(
                                "🗑️ Clear",
                                variant="secondary"
                            )
                
                script_input = gr.Textbox(
                    label="Conversation Script",
                    placeholder="Enter your podcast script here...",
                    lines=12,
                    max_lines=20,
                    elem_classes="script-input"
                )

                with gr.Row():
                    random_example_btn = gr.Button(
                        "🎲 Random Example", size="lg",
                        variant="secondary", elem_classes="random-btn", scale=1
                    )
                    generate_btn = gr.Button(
                        "🚀 Generate Podcast", size="lg",
                        variant="primary", elem_classes="generate-btn", scale=2
                    )

                gr.Markdown("### 🎵 Generated Podcast")
                complete_audio_output = gr.Audio(
                    label="Complete Podcast (Download)",
                    type="numpy",
                    elem_classes="audio-output complete-audio-section",
                    autoplay=False,
                    show_download_button=True,
                    visible=True
                )

                log_output = gr.Textbox(
                    label="Generation Log",
                    lines=8, max_lines=15,
                    interactive=False,
                    elem_classes="log-output"
                )

        # Event handlers for LLM section visibility
        def toggle_llm_settings(use_llm):
            return gr.update(visible=use_llm)

        use_llm.change(
            fn=toggle_llm_settings,
            inputs=[use_llm],
            outputs=[llm_settings]
        )

        def update_provider_visibility(provider):
            ollama_visible = provider == "ollama"
            openrouter_visible = provider == "openrouter"
            api_key_visible = provider == "openrouter"
            
            return (
                gr.update(visible=ollama_visible),
                gr.update(visible=openrouter_visible),
                gr.update(visible=api_key_visible)
            )

        llm_provider_dropdown.change(
            fn=update_provider_visibility,
            inputs=[llm_provider_dropdown],
            outputs=[ollama_model_dropdown, openrouter_model_dropdown, api_key_row]
        )

        # Save API key handler
        def save_api_key(api_key, provider, ollama_model_choice, openrouter_model_choice):
            success = True
            success &= demo_instance.config_manager.set("openrouter_api_key", api_key)
            success &= demo_instance.config_manager.set("llm_provider", provider)
            success &= demo_instance.config_manager.set("ollama_model", ollama_model_choice)
            success &= demo_instance.config_manager.set("openrouter_model", openrouter_model_choice)
            
            if success:
                return {'value': '<div class="success-message">✅ Settings saved!</div>', '__type__': 'update'}
            else:
                return {'value': '<div class="error-message">❌ Failed to save settings</div>', '__type__': 'update'}

        save_key_btn.click(
            fn=save_api_key,
            inputs=[api_key_input, llm_provider_dropdown, ollama_model_dropdown, openrouter_model_dropdown],
            outputs=[save_status]
        )

        # Update state when dropdowns change
        def update_llm_provider(provider):
            return provider

        def update_ollama_model(model):
            return model

        def update_openrouter_model(model):
            return model

        llm_provider_dropdown.change(
            fn=update_llm_provider,
            inputs=[llm_provider_dropdown],
            outputs=[llm_provider]
        )

        ollama_model_dropdown.change(
            fn=update_ollama_model,
            inputs=[ollama_model_dropdown],
            outputs=[ollama_model]
        )

        openrouter_model_dropdown.change(
            fn=update_openrouter_model,
            inputs=[openrouter_model_dropdown],
            outputs=[openrouter_model]
        )

        # Main podcast generation
        def update_speaker_visibility(num_speakers):
            return [gr.update(visible=(i < num_speakers)) for i in range(4)]

        num_speakers.change(
            fn=update_speaker_visibility,
            inputs=[num_speakers],
            outputs=speaker_selections
        )

        def generate_podcast_wrapper(model_choice, num_speakers, script, *speakers_and_params):
            try:
                speakers = speakers_and_params[:4]
                cfg_scale_val = speakers_and_params[4]
                audio, log = demo_instance.generate_podcast(
                    num_speakers=int(num_speakers),
                    script=script,
                    speaker_1=speakers[0],
                    speaker_2=speakers[1],
                    speaker_3=speakers[2],
                    speaker_4=speakers[3],
                    cfg_scale=cfg_scale_val,
                    model_name=model_choice
                )
                return audio, log
            except Exception as e:
                traceback.print_exc()
                return None, f"❌ Error: {str(e)}"

        generate_btn.click(
            fn=generate_podcast_wrapper,
            inputs=[model_dropdown, num_speakers, script_input] + speaker_selections + [cfg_scale],
            outputs=[complete_audio_output, log_output],
            queue=True
        )

        # LLM script generation
        def generate_script_with_llm(topic, num_speakers, provider, api_key, ollama_model_choice, openrouter_model_choice):
            if not topic.strip():
                return "Please enter a topic for the LLM to generate a script."
            
            if provider == "openrouter" and not api_key.strip():
                return "Error: OpenRouter API key is required. Please enter your API key in the settings above."
            
            gr.Info(f"Generating script with {provider}... This may take a moment.")
            
            model_choice = ollama_model_choice if provider == "ollama" else openrouter_model_choice
            script = demo_instance.call_llm_api(topic, int(num_speakers), provider, api_key, model_choice)
            
            if script.startswith("Error:"):
                gr.Warning(f"LLM generation failed: {script}")
            
            return script

        generate_script_btn.click(
            fn=generate_script_with_llm,
            inputs=[
                topic_input, 
                num_speakers, 
                llm_provider_dropdown,
                api_key_input,
                ollama_model_dropdown,
                openrouter_model_dropdown
            ],
            outputs=[script_input],
            queue=True
        )

        def clear_topic():
            return ""

        clear_topic_btn.click(
            fn=clear_topic,
            inputs=[],
            outputs=[topic_input]
        )

        def load_random_example():
            import random
            examples = getattr(demo_instance, "example_scripts", [])
            if not examples:
                examples = [
                    [2, "Speaker 0: Welcome to our AI podcast demo!\nSpeaker 1: Thanks, excited to be here!"]
                ]
            num_speakers_value, script_value = random.choice(examples)
            return num_speakers_value, script_value

        random_example_btn.click(
            fn=load_random_example,
            inputs=[],
            outputs=[num_speakers, script_input],
            queue=False
        )

        gr.Markdown("### 📚 Example Scripts")
        examples = getattr(demo_instance, "example_scripts", []) or [
            [1, "Speaker 1: Welcome to our AI podcast demo. This is a sample script."]
        ]
        gr.Examples(
            examples=examples,
            inputs=[num_speakers, script_input],
            label="Try these example scripts:"
        )

    return interface




def run_demo(
    model_paths: dict = None,
    device: str = "cuda",
    inference_steps: int = 5,
    share: bool = True,
):
    """
    model_paths default includes two entries. Replace paths as needed.
    """
    if model_paths is None:
        model_paths = {
            "VibeVoice-Large":"aoi-ot/VibeVoice-Large", # "microsoft/VibeVoice-Large",
            "VibeVoice-7B": "aoi-ot/VibeVoice-7B",
            "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B"
        }

    set_seed(42)
    demo_instance = VibeVoiceDemo(model_paths, device, inference_steps)
    interface = create_demo_interface(demo_instance)
    interface.queue().launch(
        share=share,
        server_name="127.0.0.1" if share else "127.0.0.1",
        show_error=True,
        show_api=False
    )



if __name__ == "__main__":
    run_demo()
