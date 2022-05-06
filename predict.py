import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datasets import ECGSequence
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str,
                        help='path to csv file containing annotations')
    parser.add_argument('path_to_model',  # or model_date_order.hdf5
                        help='file containing training model.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_file', default="./dnn_output.npy",  # or predictions_date_order.csv
                        help='output csv file.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    # Import data
    # seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs, path_to_csv=args.path_to_csv)
    seq = ECGSequence.get_test(args.path_to_hdf5, args.dataset_name, batch_size=args.bs, path_to_csv=args.path_to_csv)
    # Import model
    model = load_model(args.path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['binary_accuracy'])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    y_score = model.predict(seq,  verbose=1)
    print(y_score)
    print()
    print(len(y_score))
    preds = model.evaluate(seq, verbose=1)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    # Generate dataframe
    np.save(args.output_file, y_score)

    print("Output predictions saved")
