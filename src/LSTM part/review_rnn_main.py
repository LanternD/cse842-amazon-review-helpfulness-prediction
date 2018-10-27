from ReviewRNNProcessor import DataPreprocessor
from ReviewRNNProcessor import ReviewRNNModel
from ReviewRNNProcessor import DataPreprocessorV2
from ReviewRNNProcessor import DataPreprocessorReg
from ReviewRNNProcessor import ReviewRNNModelReg
import time


def run(function_flag):
    start_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print('Task begins. \nTime stamp: ' + current_time)

    if function_flag == 0:
        # divide the large json file into multiple small json files
        # require 'Electronics_5.json' file existing
        # build the word2vec 100-dimensional dictionary
        dpp0 = DataPreprocessor()
        dpp0.file_chunking()
        for i in range(26):
            dpp0.load_json_raw_data(i)
        dpp0.build_w2v_dict()
    elif function_flag == 1:
        # build dictionary for indexing
        for i in range(26):
            dpp0 = DataPreprocessor()
            dpp0.load_json_raw_data(i)
            dpp0.vocabulary_building()
            del dpp0
    elif function_flag == 2:
        # sequence mapping, convert the sentences into indices matrix
        dpp2 = DataPreprocessorV2()
        # for i in range(3):
        #     dpp2.chunk_json_raw_to_csv(i)
        for i in range(1):
            dpp2.load_train_test_seq(i)
        rrm0 = ReviewRNNModel(start_time, 'rnn_weights_lstm_v1')
        rrm0.network_clf()
        rrm0.model_fitting(dpp2.X_train, dpp2.Y_train, dpp2.X_test, dpp2.Y_test)

        # rrm0.knn_method(dpp2.X_train, dpp2.Y_train, dpp2.X_test, dpp2.Y_test)
        # rrm0.svm_method(dpp2.X_train, dpp2.Y_train, dpp2.X_test, dpp2.Y_test)
    elif function_flag == 3:
        # network fitting
        dpr0 = DataPreprocessorReg()
        # dpr0.chunk_json_raw_to_csv(0)
        dpr0.load_train_test_seq(0)
        rrmr0 = ReviewRNNModelReg(start_time, 'rnn_weights_reg_nz_penalty_v2')
        rrmr0.network_reg()
        rrmr0.model_fitting_reg(dpr0.X_train, dpr0.Y_train, dpr0.X_test, dpr0.Y_test)

    elif function_flag == 4:
        dpp0 = DataPreprocessor()
        for i in range(26):
            dpp0.load_json_raw_data(i)
        dpp0.tokenization()
    elif function_flag == 5:
        dpp0 = DataPreprocessor()
        dpp0.embedding_matrix_build()
    elif function_flag == 6:
        dpp2 = DataPreprocessorV2()
        for i in range(1, 26):
            dpp2.chunk_json_raw_to_csv(i)
        # dpp2.load_train_test_seq()

    end_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))

    print(current_time)
    print('Total execution time: ' + '%.3f' % (end_time - start_time) + ' s')

if __name__ == '__main__':
    run(2)
