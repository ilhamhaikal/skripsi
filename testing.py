import torch
import utils
from PIL import Image
import config as conf
import modelCRNN.crnnNet as crnnNet

# nama_file masukin nama file model
model_path = 'modelCRNN/list_model/history/epoch 20/modelCRNN_19_0.27404545454545454.pth'
#model_path = './modelCRNN/crnn.pth'
test_annotation = './90kDICT32px/annotation_test.txt'
alphabet = conf.NUMBER + conf.ALPHABET
nClass = len(alphabet) + 1
model = crnnNet.CRNN(nClass=nClass)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

converter = utils.strLabelConverter(alphabet)

transform = utils.resizeNormalize((conf.IMG_W, conf.IMG_H))

n_correct = n_total = 0
# jummlah gambar yang di test
n_stop = 599

model.eval()
with open(test_annotation, 'r') as file:
    for i, line in enumerate(file, start=1):
        parts = line.split()
        img, img_uuid = parts[0], parts[1]
        img_path = "./90kDICT32px/" + img.replace("./", "/")
        label = img_path.split(img_uuid + '.jpg')[0].split('_')[1]

        image = Image.open(img_path).convert('L')
        image = transform(image)
        image = image.view(1, *image.size())

        preds = model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = torch.IntTensor([preds.size(0)])
        pred = converter.decode(preds.data, preds_size.data, raw=False)
        if pred == label:
            n_correct += 1
        n_total += 1
        if i == n_stop:
            break

print("Jumlah benar => ", n_correct)
print("Jumlah salah => ", n_total - n_correct)
print("Persentase benar => ", (n_correct / n_total) * 100)
