import os
import sys
from typing import Iterable, Optional, Tuple, Dict, Any, List
import time
import spaces
import gradio as gr
from io import BytesIO
from PIL import Image
from loguru import logger
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import fitz  # PyMuPDF
import html2text
import markdown
import tempfile

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# --- Theme and CSS Definition ---

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8", c100="#D3E5F0", c200="#A8CCE1", c300="#7DB3D2",
    c400="#529AC3", c500="#4682B4", c600="#3E72A0", c700="#36638C",
    c800="#2E5378", c900="#264364", c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue, secondary_hue=secondary_hue, neutral_hue=neutral_hue,
            text_size=text_size, font=font, font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

# --- Model and App Logic ---

pdf_suffixes = [".pdf"]
image_suffixes = [".png", ".jpeg", ".jpg"]
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Using device: {device}")

# Model 1: Logics-Parsing
MODEL_ID_1 = "Logics-MLLM/Logics-Parsing"
logger.info(f"Loading model 1: {MODEL_ID_1}")
processor_1 = AutoProcessor.from_pretrained(MODEL_ID_1, trust_remote_code=True)
model_1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_1,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()
logger.info(f"Model '{MODEL_ID_1}' loaded successfully.")

# Model 2: Gliese-OCR-7B-Post1.0
MODEL_ID_2 = "prithivMLmods/Gliese-OCR-7B-Post1.0"
logger.info(f"Loading model 2: {MODEL_ID_2}")
processor_2 = AutoProcessor.from_pretrained(MODEL_ID_2, trust_remote_code=True)
model_2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_2,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()
logger.info(f"Model '{MODEL_ID_2}' loaded successfully.")

# Model 3: olmOCR-7B-0825
MODEL_ID_3 = "allenai/olmOCR-7B-0825"
logger.info(f"Loading model 3: {MODEL_ID_3}")
processor_3 = AutoProcessor.from_pretrained(MODEL_ID_3, trust_remote_code=True)
model_3 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_3,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()
logger.info(f"Model '{MODEL_ID_3}' loaded successfully.")

@spaces.GPU
def parse_page(image: Image.Image, model_name: str) -> str:
    if model_name == "Logics-Parsing":
        current_processor, current_model = processor_1, model_1
    elif model_name == "Gliese-OCR-7B-Post1.0":
        current_processor, current_model = processor_2, model_2
    elif model_name == "olmOCR-7B-0825":
        current_processor, current_model = processor_3, model_3
    else:
        raise ValueError(f"Unknown model choice: {model_name}")

    # Standard Qwen2-VL format
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Parse this document page into a clean, structured HTML representation. Preserve the logical structure with appropriate tags for content blocks such as paragraphs (<p>), headings (<h1>-<h6>), tables (<table>), figures (<figure>), formulas (<formula>), and others. Include category tags, and filter out irrelevant elements like headers and footers."}]}]
    
    prompt_full = current_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = current_processor(text=prompt_full, images=[image.convert("RGB")], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = current_model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    
    generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
    output_text = current_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text

def convert_file_to_images(file_path: str, dpi: int = 200) -> List[Image.Image]:
    images = []
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in image_suffixes:
        images.append(Image.open(file_path).convert("RGB"))
        return images
        
    if file_ext not in pdf_suffixes:
        raise ValueError(f"Unsupported file type: {file_ext}")

    try:
        pdf_document = fitz.open(file_path)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            images.append(Image.open(BytesIO(img_data)).convert("RGB"))
        pdf_document.close()
    except Exception as e:
        logger.error(f"Failed to convert PDF using PyMuPDF: {e}")
        raise
    return images

def get_initial_state() -> Dict[str, Any]:
    return {"pages": [], "total_pages": 0, "current_page_index": 0, "page_results": []}

def load_and_preview_file(file_path: Optional[str]) -> Tuple[Optional[Image.Image], str, Dict[str, Any]]:
    state = get_initial_state()
    if not file_path:
        return None, '<div class="page-info">No file loaded</div>', state

    try:
        pages = convert_file_to_images(file_path)
        if not pages:
            return None, '<div class="page-info">Could not load file</div>', state
        
        state["pages"] = pages
        state["total_pages"] = len(pages)
        page_info_html = f'<div class="page-info">Page 1 / {state["total_pages"]}</div>'
        return pages[0], page_info_html, state
    except Exception as e:
        logger.error(f"Failed to load and preview file: {e}")
        return None, '<div class="page-info">Failed to load preview</div>', state

async def process_all_pages(state: Dict[str, Any], model_choice: str, progress=gr.Progress(track_tqdm=True)):
    if not state or not state["pages"]:
        error_msg = "<h3>Please upload a file first.</h3>"
        return error_msg, "", "", None, "Error: No file to process", state

    logger.info(f'Processing {state["total_pages"]} pages with model: {model_choice}')
    start_time = time.time()
    
    try:
        page_results = []
        for i, page_img in progress.tqdm(enumerate(state["pages"]), desc="Processing Pages"):
            html_result = parse_page(page_img, model_choice)
            page_results.append({'raw_html': html_result})
        
        state["page_results"] = page_results
        
        full_html_content = "\n\n".join([f'<!-- Page {i+1} -->\n{res["raw_html"]}' for i, res in enumerate(page_results)])
        full_markdown = html2text.html2text(full_html_content)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(full_markdown)
            md_path = f.name
            
        parsing_time = time.time() - start_time
        cost_time_str = f'Total processing time: {parsing_time:.2f}s'
        
        current_page_results = get_page_outputs(state)
        
        return *current_page_results, md_path, cost_time_str, state

    except Exception as e:
        logger.error(f"Parsing failed: {e}", exc_info=True)
        error_html = f"<h3>An error occurred during processing:</h3><p>{str(e)}</p>"
        return error_html, "", "", None, f"Error: {str(e)}", state

def navigate_page(direction: str, state: Dict[str, Any]):
    if not state or not state["pages"]:
        return None, '<div class="page-info">No file loaded</div>', *get_page_outputs(state), state

    current_index = state["current_page_index"]
    total_pages = state["total_pages"]
    
    if direction == "prev":
        new_index = max(0, current_index - 1)
    elif direction == "next":
        new_index = min(total_pages - 1, current_index + 1)
    else:
        new_index = current_index
        
    state["current_page_index"] = new_index
    
    image_preview = state["pages"][new_index]
    page_info_html = f'<div class="page-info">Page {new_index + 1} / {total_pages}</div>'
    
    page_outputs = get_page_outputs(state)
    
    return image_preview, page_info_html, *page_outputs, state

def get_page_outputs(state: Dict[str, Any]) -> Tuple[str, str, str]:
    if not state or not state.get("page_results"):
        return "<h3>Process the document to see results.</h3>", "", ""

    index = state["current_page_index"]
    if index >= len(state["page_results"]):
        return "<h3>Result not available for this page.</h3>", "", ""
        
    result = state["page_results"][index]
    raw_html = result['raw_html']
    
    md_source = html2text.html2text(raw_html)
    md_render = markdown.markdown(md_source, extensions=['fenced_code', 'tables'])
    
    return md_render, md_source, raw_html

def clear_all():
    return None, None, "<h3>Results will be displayed here after processing.</h3>", "", "", None, "", '<div class="page-info">No file loaded</div>', get_initial_state()

css = """
.main-container { max-width: 1400px; margin: 0 auto; }
.header-text { text-align: center; margin-bottom: 20px; }
.page-info { text-align: center; padding: 8px 16px; font-weight: bold; margin: 10px 0; }
"""

with gr.Blocks() as demo:
    app_state = gr.State(value=get_initial_state())

    gr.HTML("""
    <div class="header-text">
        <h1>📄 Multimodal: VLM Parsing</h1>
        <p style="font-size: 1.1em;">An advanced Vision Language Model to parse documents and images into clean Markdown (html)</p>
        <div style="display: flex; justify-content: center; gap: 20px; margin: 15px 0;">
            <a href="https://huggingface.co/collections/prithivMLmods/mm-vlm-parsing-68e33e52bfb9ae60b50602dc" target="_blank" style="text-decoration: none; font-weight: 500;">🤗 Model Info</a>
            <a href="https://github.com/PRITHIVSAKTHIUR/VLM-Parsing" target="_blank" style="text-decoration: none; font-weight: 500;">💻 GitHub</a>
            <a href="https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending" target="_blank" style="text-decoration: none; font-weight: 500;">📝 Multimodal VLMs</a>
        </div>
    </div>
    """)

    with gr.Row(elem_classes=["main-container"]):
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(choices=["Logics-Parsing", "Gliese-OCR-7B-Post1.0", "olmOCR-7B-0825"], label="Select Model", value="Logics-Parsing")
            file_input = gr.File(label="Upload PDF or Image", file_types=[".pdf", ".jpg", ".jpeg", ".png"], type="filepath")
                    
            image_preview = gr.Image(label="Preview", type="pil", interactive=False, height=320)
            
            with gr.Row():
                prev_page_btn = gr.Button("◀ Previous")
                page_info = gr.HTML('<div class="page-info">No file loaded</div>')
                next_page_btn = gr.Button("Next ▶")

            with gr.Accordion("Download & Details", open=False):
                output_file = gr.File(label='Download Markdown Result', interactive=False)
                cost_time = gr.Textbox(label='Time Cost', interactive=False)

            example_root = "examples"
            if os.path.exists(example_root) and os.path.isdir(example_root):
                example_files = [os.path.join(example_root, f) for f in os.listdir(example_root) if f.endswith(tuple(pdf_suffixes + image_suffixes))]
                if example_files:
                    gr.Examples(examples=example_files, inputs=file_input, label="Examples")

            process_btn = gr.Button("🚀 Process Document", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Markdown Source"):
                    md_source_output = gr.Code(language="markdown", label="Markdown Source")
                with gr.Tab("Rendered Markdown"):
                    md_render_output = gr.Markdown(label='Markdown Rendering')                        
                with gr.Tab("Generated HTML"):
                    raw_html_output = gr.Code(language="html", label="Generated HTML")

    file_input.change(fn=load_and_preview_file, inputs=file_input, outputs=[image_preview, page_info, app_state], show_progress="full")
    
    process_btn.click(fn=process_all_pages, inputs=[app_state, model_choice], outputs=[md_render_output, md_source_output, raw_html_output, output_file, cost_time, app_state], show_progress="full")

    prev_page_btn.click(fn=lambda s: navigate_page("prev", s), inputs=app_state, outputs=[image_preview, page_info, md_render_output, md_source_output, raw_html_output, app_state])
    
    next_page_btn.click(fn=lambda s: navigate_page("next", s), inputs=app_state, outputs=[image_preview, page_info, md_render_output, md_source_output, raw_html_output, app_state])

    clear_btn.click(fn=clear_all, outputs=[file_input, image_preview, md_render_output, md_source_output, raw_html_output, output_file, cost_time, page_info, app_state])

if __name__ == '__main__':    
    demo.queue()
    demo.launch(theme=steel_blue_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)
