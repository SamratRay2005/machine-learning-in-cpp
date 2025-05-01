#include <iostream>
#include <numeric>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>

using namespace std;


int n;              // #features
int m;              // #train examples
double ap = 0.01;  

// Moving progress bar function
void show_progress(size_t iter, double cost) {
    const int bar_width = 40;
    static const char anim[] = "|/-\\";
    static int state = 0;
    int pos = iter % (bar_width + 1);

    cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i == pos) cout << anim[state % 4];
        else cout << '-';
    }
    cout << "] Iter: " << iter << "  Cost: " << cost << "     " << flush;
    state++;
    this_thread::sleep_for(chrono::milliseconds(10));
}

// stable sigmoid
double sigma(const vector<double>& w, double b, const vector<double>& x)
{
    double z = inner_product(w.begin(), w.end(), x.begin(), 0.0) + b;
    if (z >= 0) {
        double e = exp(-z);
        return 1.0 / (1.0 + e);
    } else {
        double e = exp(z);
        return e / (1.0 + e);
    }
}

// clamped log‐loss
double binary_cross_entropy(vector<double>& w, double b,
                            vector<double>& x, double y)
{
    double p = sigma(w, b, x);
    const double EPS = 1e-15;
    p = max(EPS, min(1.0 - EPS, p));
    return -(y * log(p) + (1 - y) * log(1 - p));
}

double cost(vector<double>& w, double b,
            vector<vector<double>>& X, vector<double>& Y)
{
    double sum = 0;
    for (int i = 0; i < m; ++i) {
        sum += binary_cross_entropy(w, b, X[i], Y[i]);
    }
    return sum / m;
}

void update(vector<double>& w, double& b,
            vector<vector<double>>& X, vector<double>& Y)
{
    vector<double> grad_w(n, 0.0);
    double grad_b = 0.0;
    for (int i = 0; i < m; ++i) {
        double err = sigma(w, b, X[i]) - Y[i];
        grad_b += err;
        for (int j = 0; j < n; ++j)
            grad_w[j] += err * X[i][j];
    }
    for (int j = 0; j < n; ++j) {
        grad_w[j] /= m;
        w[j]   -= ap * grad_w[j];
    }
    b -= ap * (grad_b / m);
}

void logistic_reg(vector<double>& w, double& b,
                  vector<vector<double>>& X, vector<double>& Y)
{
    const size_t MAX_ITERS = 15000;
    const double ABS_TOL   = 1e-8;
    const double REL_TOL   = 1e-6;

    double prev_cost = cost(w, b, X, Y);
    size_t iter = 0;
    show_progress(iter, prev_cost);

    while (iter < MAX_ITERS) {
        update(w, b, X, Y);
        double curr_cost = cost(w, b, X, Y);
        ++iter;
        show_progress(iter, curr_cost);

        double diff = fabs(prev_cost - curr_cost);
        if (diff < ABS_TOL || diff < REL_TOL * prev_cost)
            break;
        prev_cost = curr_cost;
    }
    cout << "\n";
}

int main(int argc, char* argv[]) {
    const string filename = (argc >= 2)
                            ? argv[1]
                            : "logistic_regression_dataset.csv";

    // --- 1) Load ---
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error opening file: " << filename << "\n";
        return 1;
    }

    string line;
    getline(infile, line); // skip header

    vector<vector<double>> X_all;
    vector<double> Y_all;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string token;

        getline(ss, token, ',');              // ignore ID
        getline(ss, token, ',');
        double gender = (token == "Male");    // Male=1, Female=0
        getline(ss, token, ',');
        double age    = stod(token);
        getline(ss, token, ',');
        double salary = stod(token);
        getline(ss, token, ',');
        double purchased = stod(token);

        X_all.push_back({ gender, age, salary });
        Y_all.push_back(purchased);
    }

    // --- 2) Split 80/20 ---
    int total = X_all.size();
    vector<int> idx(total);
    iota(idx.begin(), idx.end(), 0);
    mt19937_64 rng(42);
    shuffle(idx.begin(), idx.end(), rng);

    int train_count = int(0.8 * total);
    vector<vector<double>> X_train, X_test;
    vector<double> Y_train, Y_test;
    for (int i = 0; i < total; ++i) {
        if (i < train_count) {
            X_train.push_back(X_all[idx[i]]);
            Y_train.push_back(Y_all[idx[i]]);
        } else {
            X_test.push_back(X_all[idx[i]]);
            Y_test.push_back(Y_all[idx[i]]);
        }
    }

    // --- 3) Feature scaling ---
    m = X_train.size();
    n = X_train[0].size();
    vector<double> mean(n, 0.0), stdev(n, 0.0);
    // compute mean
    for (int j = 0; j < n; ++j) {
        for (auto &x : X_train) mean[j] += x[j];
        mean[j] /= m;
    }
    // compute stddev
    for (int j = 0; j < n; ++j) {
        for (auto &x : X_train) {
            double d = x[j] - mean[j];
            stdev[j] += d*d;
        }
        stdev[j] = sqrt(stdev[j]/m + 1e-8);
    }
    // apply scaling
    for (auto &x : X_train)
        for (int j = 0; j < n; ++j)
            x[j] = (x[j] - mean[j]) / stdev[j];
    for (auto &x : X_test)
        for (int j = 0; j < n; ++j)
            x[j] = (x[j] - mean[j]) / stdev[j];

    cout << "Training on " << X_train.size()
         << " examples, testing on " << X_test.size() << ".\n";

    // --- 4) Train ---
    vector<double> w(n, 0.0);
    double b = 0.0;
    logistic_reg(w, b, X_train, Y_train);

    // --- 5) Evaluate (exactly as before) ---
    int M = X_test.size();
    vector<double> preds(M);
    for (int i = 0; i < M; ++i)
        preds[i] = sigma(w, b, X_test[i]);

    double sum_sq_err = 0, sum_abs_err = 0, sum_pct_err = 0;
    int cnt_pct = 0;
    double sum_y = 0;
    for (double v : Y_test) sum_y += v;
    double mean_y = sum_y / M;
    double ss_tot = 0;
    for (int i = 0; i < M; ++i) {
        double err = preds[i] - Y_test[i];
        sum_sq_err += err*err;
        sum_abs_err += fabs(err);
        if (Y_test[i] != 0) { sum_pct_err += fabs(err/Y_test[i]); ++cnt_pct; }
        ss_tot += (Y_test[i] - mean_y)*(Y_test[i] - mean_y);
    }

    double mse  = sum_sq_err / M;
    double rmse = sqrt(mse);
    double mae  = sum_abs_err / M;
    double mape = cnt_pct ? sum_pct_err/cnt_pct*100.0 : 0.0;
    double r2   = ss_tot ? 1.0 - sum_sq_err/ss_tot : 0.0;
    int correct = 0;
    for (int i = 0; i < M; ++i)
        if ((preds[i] >= 0.5) == int(Y_test[i])) ++correct;
    double acc = double(correct) / M * 100.0;

    cout << "Test MSE:   " << mse   << "\n";
    cout << "Test RMSE:  " << rmse  << "\n";
    cout << "Test MAE:   " << mae   << "\n";
    cout << "Test MAPE:  " << mape  << "%\n";
    cout << "Test R^2:   " << r2    << "\n";
    cout << "Accuracy:   " << acc   << "%\n";

    return 0;
}
