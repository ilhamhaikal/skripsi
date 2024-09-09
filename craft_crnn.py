# import craft functions
from modelCRAFT import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

import utils
from PIL import Image
import config as conf
import modelCRNN.crnnNet as crnnNet
import torch

from modelCRAFT.file_utils import resize_with_padding_and_save
from modelCRNN.keras.crnnNet import CRNNModel
import pickle

# set image path and export folder directory
image = './imageTest/Dimas SIM A.jpg'  # can be filepath, PIL image or numpy array
output_dir = 'outputs/'
# nama_file masukin nama file model
model_path = 'modelCRNN/list_model/modelCRNN_54_0.26975.pth'

# read image
image = read_image(image)

# load models
refine_net = load_refinenet_model(cuda=False)
craft_net = load_craftnet_model(cuda=False)

# perform prediction
prediction_result = get_prediction(
    image=image,
    craft_net=craft_net,
    refine_net=refine_net,
    text_threshold=0.1,
    link_threshold=1.1,
    low_text=0.20,
    cuda=False,
    long_size=1280
)

# export detected text regions
exported_file_paths = export_detected_regions(
    image=image,
    regions=prediction_result["boxes"],
    output_dir=output_dir,
    rectify=True
)

for i in exported_file_paths:
    resize_with_padding_and_save(i)

name = ''
jenis_sim = ''
nama = ''
ttl = ''
gol_darah_kelamin = ''
alamat = ''
pekerjaan = ''
provinsi = ''
namaInRange = True
alamatInRange = False

with open('modelCRNN/keras/characters.pkl', 'rb') as file:
    characters = pickle.load(file)

img_width = 200
img_height = 50
downsample_factor = 4
char_to_num = {char: idx + 1 for idx, char in enumerate(sorted(characters))}
num_to_char = {idx + 1: char for idx, char in enumerate(sorted(characters))}
crnn_model = CRNNModel(img_width, img_height, downsample_factor, char_to_num, num_to_char)

# crnn_model.load_pretrained_model('modelCRNN/keras/saved_model/epoch_233.h5')

for index, path in enumerate(exported_file_paths):
    alphabet = conf.NUMBER + conf.ALPHABET
    nClass = len(alphabet) + 1
    model = crnnNet.CRNN(nClass=nClass)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    converter = utils.strLabelConverter(alphabet)

    transform = utils.resizeNormalize((conf.IMG_W, conf.IMG_H))
    image = Image.open(path).convert('L')
    image = transform(image)
    image = image.view(1, *image.size())

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = torch.IntTensor([preds.size(0)])
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    # sim_pred = crnn_model.predict(path)

    if sim_pred == '1':
        namaInRange = False

    if sim_pred == '2':
        alamatInRange = False

    if index == 5:
        jenis_sim = sim_pred

    if index > 6 and namaInRange:
        namaInRange = True
        nama += ' ' + sim_pred

    if alamatInRange:
        alamat += ' '+sim_pred

    if sim_pred == '1':
        alamatInRange = True
    name = ' '.join(sim_pred)

print('Jenis Sim: '+jenis_sim)
print('Nama: '+nama)
print('Alamat: '+alamat)
print(name)


# for path in exported_file_paths:
#     alphabet = conf.NUMBER + conf.ALPHABET
#     nClass = len(alphabet) + 1
#     model = crnnNet.CRNN(nClass=nClass)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#
#     converter = utils.strLabelConverter(alphabet)
#
#     transform = utils.resizeNormalize((conf.IMG_W, conf.IMG_H))
#     image = Image.open(path).convert('L')
#     image = transform(image)
#     image = image.view(1, *image.size())
#
#     model.eval()
#     preds = model(image)
#
#     _, preds = preds.max(2)
#     preds = preds.transpose(1, 0).contiguous().view(-1)
#
#     preds_size = torch.IntTensor([preds.size(0)])
#     raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
#     sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
#     name += ' ' + sim_pred
#
# print(name)
