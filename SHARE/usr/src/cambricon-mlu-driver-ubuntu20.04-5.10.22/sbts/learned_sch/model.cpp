#include "model.hpp"

using namespace mlpack;

void LschModel::InitModel(size_t stateEmbeddingSize, int hiddenNum, int *hiddenOutSize)
{
    stateInputSize = stateEmbeddingSize;
    actionOutputSize = hiddenOutSize[hiddenNum - 1];
    for (int i = 0; i < hiddenNum; i++) {
        ffn.Add(new mlpack::Linear(hiddenOutSize[i]));
        ffn.Add<ReLU>();
    }
    ffn.Add<Softmax>();
    ffn.Reset(stateEmbeddingSize);
    networkUpdatePolicy = new ens::AdamUpdate::template Policy<arma::mat, arma::mat>(
        networkUpdater, ffn.Parameters().n_rows, ffn.Parameters().n_cols);
}

void LschModel::predict(double *stateEmbeddings, std::vector<double> &result)
{
    arma::mat inputStates(stateEmbeddings, this->stateInputSize, 1, this->copyTheData, false);
    arma::mat output;
    ffn.Predict(inputStates, output);
    result.resize(output.n_elem);
    double *source = output.memptr();
    for (int i = 0; i < result.size(); i++) {
        result[i] = source[i];
    }
}

void LschModel::update(double *stateEmbeddings, int *actionIdx, double *advantages, int batchSize)
{
    arma::mat avg_lossGradient;

    for (int i = 0; i < batchSize; i++) {
        int act = actionIdx[i];
        arma::mat lossGradient(actionOutputSize, 1);
        arma::mat result_prob;
        arma::mat ffn_grad;
        arma::mat ffn_input =
            arma::mat(stateEmbeddings + i * stateInputSize, stateInputSize, 1, copyTheData, false);
        ffn.Forward(ffn_input, result_prob);
        lossGradient(act, 0) = -1 * advantages[i] / (epsilon + result_prob(act, 0));
	/*
        for (int j = 0; j < actionOutputSize; j++) {
           lossGradient(j, 0) = lossGradient(j, 0) - entropyWeight * (1.0 + arma::log(result_prob(j,
        0) + this->epsilon)); // -=? or +=?
        }
	*/
        ffn.Backward(ffn_input, lossGradient, ffn_grad);
        avg_lossGradient = (i == 0 ? ffn_grad : avg_lossGradient + ffn_grad);
    }
    avg_lossGradient /= batchSize;

    networkUpdatePolicy->Update(ffn.Parameters(), opStepSize, avg_lossGradient);
}

CNresult CNLschModelTest(int input_size)
{
    LschModel model(0.8, 0.01, 1e-6, 0.4, true);
    int hiddenOutSize[] = {64, 32, 16, 8};
    int hiddenNum = 4;
    int batch_size = 6;
    int action_dim = hiddenOutSize[hiddenNum - 1];
    int tmp1 = input_size * batch_size;
    int tmp2 = action_dim * batch_size;
    double *train_embedding = new double[tmp1];
    double *test_embedding = new double[tmp1];
    int *idx = new int[tmp2];
    double *adv = new double[tmp2];
    std::vector<double> result;
    int tmp;

    model.InitModel(input_size, hiddenNum, hiddenOutSize);

    for (int i = 0; i < tmp1; i++) {
        train_embedding[i] = static_cast<double>(rand() / RAND_MAX);
        test_embedding[i] = static_cast<double>(rand() / RAND_MAX);
    }

    for (int i = 0; i < tmp2; i++) {
        idx[i] = static_cast<int>(rand() / RAND_MAX);
        adv[i] = static_cast<double>(rand() / RAND_MAX);
    }

    model.update(train_embedding, idx, adv, batch_size);

    model.predict(test_embedding, result);

    delete[] train_embedding;
    delete[] test_embedding;
    delete[] idx;
    delete[] adv;

    return CN_SUCCESS;
}

