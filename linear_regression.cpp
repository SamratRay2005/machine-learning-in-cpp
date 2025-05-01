#include <iostream>
#include <numeric>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>

using namespace std;

// j = (1/2m)*sum(y_i - y)^2
// w = w - alpha * (d(j)/dw)
// dj/dw =  (1/m) * sum((f - y)x)
// dj/db  =  (1/m) * sum((f - y))
// f = wx + b

int n;
int m;
double ap = 1e-5;

double dot_product(const vector<double>& a, const vector<double>& b)
{
    return inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double retf(vector<double>& w , double & b , vector<double>& x)
{
    return dot_product(w,x) + b;
}

double cost(vector<double>& w , double & b , vector<vector<double>>& x , vector<double>& y)
{
    double sum = 0;
    for (int i = 0 ; i < m ; i++)
    {
        double d = (retf(w , b , x[i]) - y[i]);
        sum += d * d;
    }
    return sum / (2 * m);
}

void up_w(vector<double>& w ,double& b ,  vector<vector<double>>& x , vector<double>& y)
{
    vector<double> sum(n, 0.0);
    for (int i = 0 ; i < m ; i++)
    {
        double mv = retf(w , b , x[i]) - y[i];
        for (int j = 0 ; j < n ; j++)
        {
            sum[j] += mv * x[i][j];
        }
    }
    for  (int i = 0 ; i < n ; i++)
    {
        sum[i] = sum[i] / m;
        w[i] -= ap * sum[i];
    }
}

void up_b(vector<double>& w , double& b , vector<vector<double>>& x , vector<double>& y)
{
    double sum = 0.0;
    for (int i = 0 ; i < m ; i++)
    {
        double mv = retf(w , b , x[i]) - y[i];
        sum += mv;
    }
    sum = sum / m;
    b -= ap * sum;
}

void linear_reg(vector<double>& w , double& b , vector<vector<double>>& x , vector<double>& y)
{
    double prev_cost = cost(w, b, x, y);

    while (true) {
        up_w(w, b, x, y);
        up_b(w, b, x, y);
        double curr_cost = cost(w, b, x, y);
        if (fabs(prev_cost - curr_cost) < 1e-8) break;
        prev_cost = curr_cost;
    }
}

vector<double> split_line(const string& line) {
    vector<double> vals;
    string token;
    stringstream ss(line);

    while (getline(ss, token, ',')) {
        if (token == "NA") {
            return {};  // early return on missing data
        }
        try {
            vals.push_back(stod(token));
        } catch (...) {
            cerr << "Failed to parse line: " << line << endl;
            return {};
        }
    }
    return vals;
}

int main() {
    constexpr char filename[] = "babies.csv";
    ifstream file(filename);
    if (!file) {
        cerr << "Failed to open file: " << filename << endl;
        return 1;
    }

    string line;
    getline(file, line); // skip header

    vector<vector<double>> x_all;
    vector<double> y_all;
    x_all.reserve(200);
    y_all.reserve(200);

    while (x_all.size() < 200 && getline(file, line)) {
        auto v = split_line(line);
        if (v.size() == 8) {
            y_all.push_back(v[1]);
            x_all.push_back({v[2], v[3], v[4], v[5], v[6], v[7]});
        }
    }
    file.close();

    size_t total = x_all.size();
    if (total == 0) {
        cerr << "No valid data loaded.\n";
        return 1;
    }
    cout << "Loaded " << total << " valid rows.\n";

    // Shuffle + split 80/20
    vector<size_t> idx(total);
    iota(idx.begin(), idx.end(), 0);
    mt19937 rng(random_device{}());
    shuffle(idx.begin(), idx.end(), rng);

    size_t split_idx = size_t(total * 0.8);
    vector<vector<double>> x_train, x_test;
    vector<double> y_train, y_test;
    for (size_t i = 0; i < split_idx; ++i) {
        x_train.push_back(x_all[idx[i]]);
        y_train.push_back(y_all[idx[i]]);
    }
    for (size_t i = split_idx; i < total; ++i) {
        x_test.push_back(x_all[idx[i]]);
        y_test.push_back(y_all[idx[i]]);
    }

    // --- FEATURE NORMALIZATION ---
    n = int(x_train[0].size());
    vector<double> mu(n,0.0), sigma(n,0.0);

    // compute mean
    for (int j = 0; j < n; ++j)
        for (auto& row: x_train) mu[j] += row[j];
    for (int j = 0; j < n; ++j) mu[j] /= x_train.size();

    // compute stddev
    for (int j = 0; j < n; ++j)
        for (auto& row: x_train)
            sigma[j] += (row[j] - mu[j])*(row[j] - mu[j]);
    for (int j = 0; j < n; ++j) {
        sigma[j] = sqrt(sigma[j] / x_train.size());
        if (sigma[j] == 0) sigma[j] = 1;  // avoid divide by zero
    }

    // normalize train
    for (auto& row: x_train)
        for (int j = 0; j < n; ++j)
            row[j] = (row[j] - mu[j]) / sigma[j];

    // normalize test using train params
    for (auto& row: x_test)
        for (int j = 0; j < n; ++j)
            row[j] = (row[j] - mu[j]) / sigma[j];

    // set m *after* split
    m = int(x_train.size());

    vector<double> w(n, 0.0);
    double b = 0.0;
    linear_reg(w, b, x_train, y_train);

    // Evaluate on test set
    size_t m_test = x_test.size();
    double mse=0, mae=0, mape=0, ss_res=0, ss_tot=0;
    double mean_y = accumulate(y_test.begin(), y_test.end(), 0.0) / m_test;
    size_t nz = 0;
    for (size_t i = 0; i < m_test; ++i) {
        double pred = retf(w,b,x_test[i]);
        double err = y_test[i] - pred;
        mse   += err*err;
        mae   += fabs(err);
        if (y_test[i]!=0){ mape += fabs(err)/fabs(y_test[i]); ++nz; }
        ss_res+= err*err;
        ss_tot+= (y_test[i]-mean_y)*(y_test[i]-mean_y);
    }
    mse /= m_test;  mae /= m_test;
    mape = (nz? (mape/nz)*100 : NAN);
    double rmse = sqrt(mse);
    double r2   = 1 - (ss_res/ss_tot);
    double acc  = 100 - mape;

    cout << "Test MSE:   " << mse   << "\n";
    cout << "Test RMSE:  " << rmse  << "\n";
    cout << "Test MAE:   " << mae   << "\n";
    cout << "Test MAPE:  " << mape  << "%\n";
    cout << "Test R^2:   " << r2    << "\n";
    cout << "Accuracy:   " << acc   << "%\n";

    return 0;
}