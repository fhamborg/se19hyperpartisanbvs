import ast
from optparse import OptionParser
from os.path import join

import numpy as np
from keras.layers import Average, Input
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder

max_len = 200


class Inferrer:
    def __init__(self, basepath_saved_models: str = 'saved_models/'):
        # got the following three values from running ensemble_pred.sh
        saved_model1 = join(basepath_saved_models, 'BNCNN_vacc0.9077_f4_e10.hdf5')
        saved_model2 = join(basepath_saved_models, 'BNCNN_vacc0.9048_f9_e02.hdf5')
        saved_model3 = join(basepath_saved_models, 'BNCNN_vacc0.8923_f4_e08.hdf5')
        model1 = load_model(saved_model1)
        model1.name = 'model1'
        model2 = load_model(saved_model2)
        model2.name = 'model2'
        model3 = load_model(saved_model3)
        model3.name = 'model3'

        models = [model1, model2, model3]
        model_input = Input(shape=models[0].input_shape[1:], dtype='float32')
        self.ensemble_models = ensemble(models, model_input)

        # print(models[0].input_shape[1:])

    def infer_from_elmorows(self, elmorows: list):
        data = []
        l = []
        ids = []
        i = 0
        l_encoder = LabelEncoder()

        for row in elmorows:
            gzip_fields = row.split('\t')  # row.split('\t')
            gzip_id = gzip_fields[0]
            gzip_label = gzip_fields[1]
            elmo_embd_str = gzip_fields[4].strip()
            elmo_embd_list = ast.literal_eval(elmo_embd_str)
            elmo_embd_array = np.array(elmo_embd_list)
            padded_seq = sequence.pad_sequences([elmo_embd_array], maxlen=max_len, dtype='float32')[0]
            data.append(padded_seq)
            l.append(gzip_label)
            ids.append(gzip_id)
            i += 1
        label = l_encoder.fit_transform(l)
        x_data = np.array(data)
        y_data = np.array(label)
        doc_ids = np.array(ids)

        preds = self.infer(x_data)

        docids_predictions = {}
        for i in range(len(doc_ids)):
            current_doc_id = doc_ids[i]
            current_prob = preds[i][0]
            # current_prob = all_prediction[i]
            if current_prob > 0.5:
                current_pred = True
            else:
                current_prob = 1 - current_prob
                current_pred = False

            docids_predictions[current_doc_id] = (current_pred, current_prob)

        return docids_predictions

    def infer(self, x_data):
        pred = self.ensemble_models.predict(x_data)
        return pred


def toEvaluationFormat(all_doc_ids, all_prediction):
    evaluationFormatList = []
    for i in range(len(all_doc_ids)):
        current_doc_id = all_doc_ids[i]
        current_prob = all_prediction[i][0]
        # current_prob = all_prediction[i]
        if current_prob > 0.5:
            current_pred = 'true'
        else:
            current_prob = 1 - current_prob
            current_pred = 'false'
        evaluationFormat = str(current_doc_id) + ' ' + str(current_pred) + ' ' + str(current_prob) + '\n'
        evaluationFormatList.append(evaluationFormat)
    return evaluationFormatList


def load_data(data_path, max_len=200):
    data = []
    l = []
    ids = []
    i = 0
    l_encoder = LabelEncoder()
    with open(data_path, 'rb') as inf:
        for line in inf:
            gzip_fields = line.decode('utf-8').split('\t')
            gzip_id = gzip_fields[0]
            gzip_label = gzip_fields[1]
            elmo_embd_str = gzip_fields[4].strip()
            elmo_embd_list = ast.literal_eval(elmo_embd_str)
            elmo_embd_array = np.array(elmo_embd_list)
            padded_seq = sequence.pad_sequences([elmo_embd_array], maxlen=max_len, dtype='float32')[0]
            data.append(padded_seq)
            l.append(gzip_label)
            ids.append(gzip_id)
            i += 1
            print(i)
    label = l_encoder.fit_transform(l)
    return np.array(data), np.array(label), np.array(ids)


def ensemble(models, model_input):
    outputs = [model(model_input) for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')

    return model


def main():
    # TODO this currently cannot be called anymore as the models that are passed from cmd will not be used during
    # Inferrer.__init__()
    parser = OptionParser()
    parser.add_option("--inputTSV", help="load saved cache", type=str, default='work/train.elmo.tsv')
    parser.add_option("--output", help="load saved cache", type=str, default='work/test.tsv')
    parser.add_option("--saved_model1", help="load saved cache", type=str)
    parser.add_option("--saved_model2", help="load saved cache", type=str)
    parser.add_option("--saved_model3", help="load saved cache", type=str)

    options, arguments = parser.parse_args()

    embed_size = 1024
    seed = 7

    inferrer = Inferrer()

    # data preparing
    x_data, y_data, doc_id = load_data(options.inputTSV, max_len=max_len)

    pred = inferrer.infer(x_data)

    all_pred = toEvaluationFormat(doc_id, pred)
    with open(options.output, 'w') as fo:
        for item in all_pred:
            fo.write(item)


if __name__ == '__main__':
    main()
