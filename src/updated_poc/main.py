from PIL import Image
from tqdm import tqdm
from smart_crop import *
import time
import random
from subprocess import getoutput, run
import shutil
import os

IMAGES_FOLDER_OPTIONAL = "./data/original/"
INSTANCE_DIR = "./data/reshaped/"
Crop_size = 512


def convert_images():
    for filename in tqdm(
        os.listdir(IMAGES_FOLDER_OPTIONAL),
        bar_format="  |{bar:15}| {n_fmt}/{total_fmt} Uploaded",
    ):
        extension = filename.split(".")[-1]
        new_path_with_file = os.path.join(INSTANCE_DIR, filename)
        file = Image.open(IMAGES_FOLDER_OPTIONAL + "/" + filename)
        width, height = file.size
        if file.size != (Crop_size, Crop_size):
            image = crop_image(file, Crop_size)
            if extension.upper() == "JPG" or extension.lower() == "jpg":
                image[0] = image[0].convert("RGB")
                image[0].save(new_path_with_file, format="JPEG", quality=100)
            else:
                image[0].save(new_path_with_file, format=extension.upper())

def train_only_unet(style: str, extrnlcptn: str, text_encoder_training_steps: int, stpsv: int, 
                    stp: int, session_directory: str, sb_model_name: str, instance_directory:str, 
                    output_directory: str, pt: str, seed:int, res: int, caption_directory: str,
                    precision: str, training_steps: int, gc: str, untlr: float):
    try:
        print('Training only UNET')
        run(
        [
            "accelerate",
            "launch",
            "./diffusers/examples/dreambooth/train_dreambooth.py",
            "--stop_text_encoder_training=" + str(text_encoder_training_steps),
            "--image_captions_filename",
            "--train_only_unet",
            "--save_starting_step=" + str(stpsv),
            "--save_n_steps=" + str(stp),
            "--Session_dir=" + str(session_directory),
            "--pretrained_model_name_or_path=" + str(sb_model_name),
            "--instance_data_dir=" + str(instance_directory),
            "--output_dir=" + str(output_directory),
            "--captions_dir=" + str(caption_directory),
            "--instance_prompt=" + str(pt),
            "--seed=" + str(seed),
            "--resolution=" + str(res),
            "--mixed_precision=" + str(precision),
            "--train_batch_size=1",
            "--max_train_steps=" + str(training_steps),
            "--gradient_accumulation_steps=1",
            # str(gc),
            "--learning_rate=" + str(untlr),
            "--use_8bit_adam",
            "--lr_scheduler=linear",
            "--lr_warmup_steps=0"
        ])
    except Exception as e:
        print(e)
        
def rename_images(src_dir, dest_dir, token):
    # Eliminar contenido del directorio de destino
    for file_name in os.listdir(dest_dir):
        file_path = os.path.join(dest_dir, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error al eliminar {file_name} en {dest_dir}. Error: {e}')
            return
    
    # Obtener todos los archivos de imagen en el directorio de origen
    src_files = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF'))]
    print('src_files = {0}'.format(src_files))
    # Crear el directorio de destino si no existe
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Recorrer todos los archivos de imagen y renombrarlos
    for i, src_file in enumerate(src_files):
        # Construir el nuevo nombre del archivo
        if i == 0:
            new_name = f'{token}.{src_file.split(".")[-1]}'
        else:
            new_name = f'{token}-({i}).{src_file.split(".")[-1]}'
        
        # Construir las rutas completas
        src_path = os.path.join(src_dir, src_file)
        dest_path = os.path.join(dest_dir, new_name)
                
        # Renombrar y mover el archivo
        shutil.copy2(src_path, dest_path)
    
def create_directory_if_not_exists(directory_path: str):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def train_model(
    *,
    training_subject: str = "Character", # @param ["Character", "Object", "Style", "Artist", "Movie", "TV Show"] 
    with_prior_preservation: bool = True, #@param ["Yes", "No"] 
    # With the prior reservation method, the results are better, you will either have to upload around 200 
    # pictures of the class you're training (dog, person, car, house ...) or let Dreambooth generate them.
    model_name: str = "stable-diffusion-v2-512",
    subject_type: str = "person", #@param{type: 'string'}
    # If you're training on a character or an object, the subject type would be : Man, Woman, Shirt, Car, Dog, Baby ...etc
    # If you're training on a Style, the subject type would be : impressionist, brutalist, abstract, use "beautiful" for a general style...etc
    # If you're training on a Movie/Show, the subject type would be : Action, Drama, Science-fiction, Comedy ...etc
    # If you're training on an Artist, the subject type would be : Painting, sketch, drawing, photography, art ...etc
    instance_name: str = "daniel_alpiste", #@param{type: 'string'}
    # The instance is an identifier, choose a unique identifier unknown by stable diffusion. 
    subject_images_number: int = 500, #@param{type: 'number'}
    # The number of images you want to train on, the more the better, but it will take longer to train.
    unet_training_steps: int = 1500, # @param{type: 'number'}
    unet_learning_rate: float = 2e-6, # @param ["2e-5","1e-5","9e-6","8e-6","7e-6","6e-6","5e-6", "4e-6", "3e-6", "2e-6"] {type:"raw"}
    text_encoder_training_steps: int = 250,
    text_encoder_concept_training_steps: int = 0,
    text_encoder_learning_rate: float = 1e-6,
    offset_noise: bool = None,
    external_captions: bool = False, # @param {type:"boolean"}
    # @markdown - Get the captions from a text file for each instance image.
    res: int = 512, # @param ["256", "512", "1024"] {type:"raw"}
    fp16: bool = True,
    enable_text_encoder_training: bool = True,
    enable_text_encoder_concept_training: bool = True,
    save_checkpoint_every_n_steps: bool = False, # @param {type:"boolean"}
    save_checkpoint_every: int = 500,
    start_saving_from_the_step: int = 500,
    disconnect_after_training: bool = False, # @param {type:"boolean"}
    safetensors: bool = False, # @param {type:"boolean"}
):
    modelnm=""
    sftnsr=""
    if not safetensors:
        modelnm="model.ckpt"
    else:
        modelnm="model.safetensors"
        sftnsr="--from_safetensors"
  
    if training_subject == "Character" or training_subject == "Object":
        PT = "photo of "+ instance_name + " " + subject_type
        CPT = "a photo of a "+ instance_name +", ultra detailed"
    
    style_training = False
    if training_subject == "Style":
        style_training = True
    
    output_directory = os.path.join('./data/models/', instance_name)

    # [TODO] !mkdir -p "$INSTANCE_DIR" -> Funcion para hacer la carpeta de la instancia pero en este caso ya esta hecha
    
    PT = subject_type
    session_name = instance_name
    session_directory = os.path.join('./data/sessions/', session_name)
    instance_directory_session = os.path.join(session_directory, 'instance_images')
    concept_directory_session = os.path.join(session_directory, 'concept_images')
    captions_directory_session = os.path.join(session_directory, 'captions')
    mdlpth = str(os.path.join(session_directory, session_name+'.ckpt'))
    directories_list = [output_directory, session_directory, instance_directory_session, concept_directory_session, captions_directory_session]

    for directory in directories_list:
        create_directory_if_not_exists(directory)

    # cd ./data (?) Why? I will try discovering it later.  
    # Insert the photos of the github repo in the concept_directory_session
    # dataset="person_ddim" #@param ["man_euler", "man_unsplash", "person_ddim", "woman_ddim", "blonde_woman"]
    
    # !git clone https://github.com/djbielejeski/Stable-Diffusion-Regularization-Images-{dataset}.git
    # if not os.path.exists(str(CONCEPT_DIR)):
    #   %mkdir -p "$CONCEPT_DIR"
    # %mkdir -p regularization_images/{dataset}
    # %mv -v Stable-Diffusion-Regularization-Images-{dataset}/{dataset}/*.* "$CONCEPT_DIR"
    # CLASS_DIR=CONCEPT_DIR

    rename_images("./data/reshaped/" + instance_name, instance_directory_session, instance_name)
    
    if os.path.exists(instance_directory_session+"/.ipynb_checkpoints"):
        shutil.rmtree(os.path.join(instance_directory_session, ".ipynb_checkpoints"))

    if os.path.exists(concept_directory_session+"/.ipynb_checkpoints"):
        shutil.rmtree(os.path.join(concept_directory_session, ".ipynb_checkpoints"))

    if os.path.exists(captions_directory_session+"/.ipynb_checkpoints"):
        shutil.rmtree(os.path.join(captions_directory_session, ".ipynb_checkpoints"))
        
    trnonltxt = ""
    if unet_training_steps == 0:
        trnonltxt = "--train_only_text_encoder"
    
    seed = random.randint(1, 999999)

    extrnlcptn = ""
    if external_captions:
        extrnlcptn = "--external_captions"

    style_var=""
    if style_training:
        style_var="--Style"

    if fp16:
        prec="fp16"
    else:
        prec="no"
    
    gc ="--gradient_checkpointing"
    s_nvidia = getoutput('nvidia-smi')
    if 'A100' in s_nvidia:
        gc=""

    resuming = ""

    if text_encoder_training_steps == 0 or external_captions:
        enable_text_encoder_training = False
    else:
        stptxt = text_encoder_training_steps

    if text_encoder_concept_training_steps == 0:
        enable_text_encoder_concept_training = False
    else:
        stptxtc = text_encoder_concept_training_steps

    if save_checkpoint_every == None:
        save_checkpoint_every = 1

    stp = 0
    
    if start_saving_from_the_step == None:
        start_saving_from_the_step = 0
    if (start_saving_from_the_step < 200):
        start_saving_from_the_step = save_checkpoint_every
    stpsv = start_saving_from_the_step
    
    if save_checkpoint_every_n_steps:
        stp = save_checkpoint_every


    train_only_unet(style=style_var,
                    extrnlcptn=extrnlcptn,
                    text_encoder_training_steps=text_encoder_training_steps,
                    stpsv=stpsv, 
                    stp=stp,
                    session_directory=session_directory,
                    sb_model_name=model_name,
                    instance_directory=instance_directory_session,
                    output_directory=output_directory,
                    pt=subject_type,
                    seed=seed,
                    res=res,
                    caption_directory=captions_directory_session,
                    precision=prec, 
                    training_steps=unet_training_steps,
                    gc=gc,
                    untlr=unet_learning_rate
                    )


# convert_images()
train_model()
