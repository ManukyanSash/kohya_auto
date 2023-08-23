import os
import shutil
import gradio as gr
from easygui import msgbox
from itertools import chain
from collections import namedtuple
from .common_gui import get_folder_path, add_pre_postfix, find_replace
from .basic_caption_gui import caption_images as basic_captioning
from .blip_caption_gui import caption_images as blip_captioning
from .git_caption_gui import caption_images as git_captioning
from .wd14_caption_gui import caption_images as wd14_captioning

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

def creating_folder_and_mv_images_and_captions(path, new_folder_name):
    prev_path = os.getcwd()
    os.chdir(path)
    destination_dir = './' + new_folder_name
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    txt_files = [file for file in os.listdir(path) if file.endswith(".txt")]

    img_ext_list = ['.jpg', '.jpeg', '.png']

    for filename in os.listdir(path):
        source_path = os.path.join(path, filename)
        destination_path = os.path.join(destination_dir, filename)
    
        # Check if the file has an image extension
        if any(filename.lower().endswith(ext) for ext in img_ext_list):
            shutil.copy(source_path, destination_path)

    for txt_file in txt_files:
        source_path = os.path.join(path, txt_file)
        destination_path = os.path.join(destination_dir, txt_file)
        shutil.move(source_path, destination_path)

    os.chdir(prev_path)

# Gradio UI
def gradio_all_caption_gui_tab(headless=False):
    with gr.Tab('All Captioning'):
        train_data_dir = gr.Textbox(
                label='Image folder to caption',
                placeholder='Directory containing the images to caption',
                interactive=True,
            )
        button_train_data_dir_input = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
        button_train_data_dir_input.click(
                get_folder_path,
                outputs=train_data_dir,
                show_progress=False,
            )
        gr.Markdown(
            'Basic Captioning'
            )
        with gr.Row():
            caption_ext = gr.Textbox(
                label='Caption file extension',
                placeholder='Extension for caption file (e.g., .caption, .txt)',
                value='.txt',
                interactive=True,
            )
            overwrite = gr.Checkbox(
                label='Overwrite existing captions in folder',
                interactive=True,
                value=False,
            )
        with gr.Row():
            prefix = gr.Textbox(
                label='Prefix to add to caption',
                placeholder='(Optional)',
                interactive=True,
            )
            caption_text = gr.Textbox(
                label='Caption text',
                placeholder='e.g., "by some artist". Leave empty if you only want to add a prefix or postfix.',
                interactive=True,
            )
            postfix = gr.Textbox(
                label='Postfix to add to caption',
                placeholder='(Optional)',
                interactive=True,
            )
        with gr.Row():
            find_text = gr.Textbox(
                label='Find text',
                placeholder='e.g., "by some artist". Leave empty if you only want to add a prefix or postfix.',
                interactive=True,
            )
            replace_text = gr.Textbox(
                label='Replacement text',
                placeholder='e.g., "by some artist". Leave empty if you want to replace with nothing.',
                interactive=True,
            )
        basic_inputs = [
            caption_text,
            train_data_dir,
            overwrite,
            caption_ext,
            prefix,
            postfix,
            find_text,
            replace_text]
            
        ###############

        gr.Markdown(
            'Blip captioning'
        )
        with gr.Row():
            caption_file_ext = gr.Textbox(
                label='Caption file extension',
                placeholder='Extension for caption file, e.g., .caption, .txt',
                value='.txt',
                interactive=True,
            )

            prefix = gr.Textbox(
                label='Prefix to add to BLIP caption',
                placeholder='(Optional)',
                interactive=True,
            )

            postfix = gr.Textbox(
                label='Postfix to add to BLIP caption',
                placeholder='(Optional)',
                interactive=True,
            )

            batch_size = gr.Number(
                value=1, label='Batch size', interactive=True
            )

        with gr.Row():
            beam_search = gr.Checkbox(
                label='Use beam search', interactive=True, value=True
            )
            num_beams = gr.Number(
                value=1, label='Number of beams', interactive=True
            )
            top_p = gr.Number(value=0.9, label='Top p', interactive=True)
            max_length = gr.Number(
                value=75, label='Max length', interactive=True
            )
            min_length = gr.Number(
                value=5, label='Min length', interactive=True
            )

        blip_inputs = [
            train_data_dir,
            caption_file_ext,
            batch_size,
            num_beams,
            top_p,
            max_length,
            min_length,
            beam_search,
            prefix,
            postfix
        ]
        
        ############
        
        gr.Markdown(
            'GIT captioning'
        )

        with gr.Row():
            caption_ext = gr.Textbox(
                label='Caption file extension',
                placeholder='Extention for caption file. eg: .caption, .txt',
                value='.txt',
                interactive=True,
            )

            prefix = gr.Textbox(
                label='Prefix to add to BLIP caption',
                placeholder='(Optional)',
                interactive=True,
            )

            postfix = gr.Textbox(
                label='Postfix to add to BLIP caption',
                placeholder='(Optional)',
                interactive=True,
            )

            batch_size = gr.Number(
                value=1, label='Batch size', interactive=True
            )

        with gr.Row():
            max_data_loader_n_workers = gr.Number(
                value=2, label='Number of workers', interactive=True
            )
            max_length = gr.Number(
                value=75, label='Max length', interactive=True
            )
            model_id = gr.Textbox(
                label='Model',
                placeholder='(Optional) model id for GIT in Hugging Face',
                interactive=True,
            )

        git_inputs = [
            train_data_dir,
            caption_ext,
            batch_size,
            max_data_loader_n_workers,
            max_length,
            model_id,
            prefix,
            postfix
        ]
        
        ############

        gr.Markdown(
            'WD14 captioning'
        )

        # Input Settings
        # with gr.Section('Input Settings'):
        with gr.Row():
            caption_extension = gr.Textbox(
                label='Caption file extension',
                placeholder='Extention for caption file. eg: .caption, .txt',
                value='.txt',
                interactive=True,
            )

        undesired_tags = gr.Textbox(
            label='Undesired tags',
            placeholder='(Optional) Separate `undesired_tags` with comma `(,)` if you want to remove multiple tags, e.g. `1girl,solo,smile`.',
            interactive=True,
        )

        with gr.Row():
            prefix = gr.Textbox(
                label='Prefix to add to WD14 caption',
                placeholder='(Optional)',
                interactive=True,
            )

            postfix = gr.Textbox(
                label='Postfix to add to WD14 caption',
                placeholder='(Optional)',
                interactive=True,
            )

        with gr.Row():
            replace_underscores = gr.Checkbox(
                label='Replace underscores in filenames with spaces',
                value=True,
                interactive=True,
            )
            recursive = gr.Checkbox(
                label='Recursive',
                value=False,
                info='Tag subfolders images as well',
            )

            debug = gr.Checkbox(
                label='Verbose logging',
                value=True,
                info='Debug while tagging, it will print your image file with general tags and character tags.',
            )
            frequency_tags = gr.Checkbox(
                label='Show tags frequency',
                value=True,
                info='Show frequency of tags for images.',
            )

        # Model Settings
        with gr.Row():
            model = gr.Dropdown(
                label='Model',
                choices=[
                    'SmilingWolf/wd-v1-4-convnext-tagger-v2',
                    'SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
                    'SmilingWolf/wd-v1-4-vit-tagger-v2',
                    'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
                ],
                value='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
            )

            general_threshold = gr.Slider(
                value=0.35,
                label='General threshold',
                info='Adjust `general_threshold` for pruning tags (less tags, less flexible)',
                minimum=0,
                maximum=1,
                step=0.05,
            )
            character_threshold = gr.Slider(
                value=0.35,
                label='Character threshold',
                info='useful if you want to train with character',
                minimum=0,
                maximum=1,
                step=0.05,
            )

        # Advanced Settings
        with gr.Row():
            batch_size = gr.Number(
                value=8, label='Batch size', interactive=True
            )

            max_data_loader_n_workers = gr.Number(
                value=2, label='Max dataloader workers', interactive=True
            )

        wd14_inputs = [
            train_data_dir,
            caption_extension,
            batch_size,
            general_threshold,
            character_threshold,
            replace_underscores,
            model,
            recursive,
            max_data_loader_n_workers,
            debug,
            undesired_tags,
            frequency_tags,
            prefix,
            postfix
        ]

        model_info = namedtuple("Model", ["name", "function", "inputs"])

        models_list = [model_info(name="basic", function=basic_captioning, inputs=basic_inputs), 
                       model_info(name="blip", function=blip_captioning, inputs=blip_inputs), 
                       model_info(name="git", function=git_captioning, inputs=git_inputs), 
                       model_info(name="wd14", function=wd14_captioning, inputs=wd14_inputs)]
        
        def all_captionings(*args):
            offset = 0
            for model in models_list:
                model.function(*args[offset:len(model.inputs) + offset])
                creating_folder_and_mv_images_and_captions(args[1], model.name)
                offset += len(model.inputs)

        caption_button = gr.Button('Caption images')

        caption_button.click(
            all_captionings,
            inputs = list(chain(*list(model.inputs for model in models_list))),
            show_progress = False
        )
