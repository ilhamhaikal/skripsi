import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from PIL import Image
import config as conf
import modelCRNN.crnnNet as crnnNet

# nama_file masukin nama file model
model_path = 'modelCRNN/list_model/history/epoch 30/modelCRNN_25_0.27238636363636365.pth'
#model_path = './modelCRNN/crnn.pth'
img_path = './imageTest/6389.png'
alphabet = conf.NUMBER + conf.ALPHABET
nClass = len(alphabet) + 1
model = crnnNet.CRNN(nClass=nClass)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

converter = utils.strLabelConverter(alphabet)

transform = utils.resizeNormalize((conf.IMG_W, conf.IMG_H))
image = Image.open(img_path).convert('L')
image = transform(image)
image = image.view(1, *image.size())

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = torch.IntTensor([preds.size(0)])
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
