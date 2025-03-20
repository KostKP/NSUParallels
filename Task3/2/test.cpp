#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <cmath>
#include <vector>
#include <sstream>
#include <iomanip>
#include <variant>
#include <optional>

using namespace std;

struct OneArgTask {
    double arg;
};

struct TwoArgsTask {
    float arg1;
    float arg2;
};

using TaskArgs = variant<OneArgTask, TwoArgsTask>;

struct TaskInfo {
    size_t id;
    TaskArgs args;
    double result;
};

enum class OperationType {
    POW,
    SIN,
    SQRT
};

OperationType getOperationType(const string& filename) {
    if (filename.find("pow") != string::npos) return OperationType::POW;
    if (filename.find("sin") != string::npos) return OperationType::SIN;
    if (filename.find("sqrt") != string::npos) return OperationType::SQRT;
    throw runtime_error("Unknown file type: " + filename);
}

optional<TaskInfo> parseLine(const string& line, OperationType opType) {
    TaskInfo task;
    if (opType == OperationType::POW) {
        TwoArgsTask args;
        if (sscanf(line.c_str(), "[%zu] (%f, %f) => %lf", 
                  &task.id, &args.arg1, &args.arg2, &task.result) != 4)
            return nullopt;
        task.args = args;
    } else {
        OneArgTask args;
        if (sscanf(line.c_str(), "[%zu] (%lf) => %lf", 
                  &task.id, &args.arg, &task.result) != 3)
            return nullopt;
        task.args = args;
    }
    return task;
}

double calculateExpected(const TaskInfo& task, OperationType opType) {
    return visit([opType](auto&& arg) -> double {
        using T = decay_t<decltype(arg)>;

        if constexpr (is_same_v<T, OneArgTask>) {
            switch (opType) {
                case OperationType::SIN: return sin(arg.arg);
                case OperationType::SQRT: return sqrt(arg.arg);
                default: return NAN;
            }
        } else if constexpr (is_same_v<T, TwoArgsTask>) {
            return pow(static_cast<double>(arg.arg1), static_cast<double>(arg.arg2));
        }
        return NAN;
    }, task.args);
}

bool validateResult(double expected, double actual) {
    const double relTolerance = 0.001;  // Current tolerance
    const double absTolerance = 0.0001;  // Absolute tolerance for very small values

    if (isnan(expected)) return false;

    return abs(expected - actual) < max(relTolerance * max(abs(expected), abs(actual)), absTolerance);
}

bool processFile(const string& filename, set<size_t>& globalIds) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "  Error opening file: " << filename << endl;
        return false;
    }

    OperationType opType = getOperationType(filename);
    set<size_t> localIds;
    string line;
    bool valid = true;

    for (size_t lineNum = 1; getline(file, line); ++lineNum) {
        auto task = parseLine(line, opType);
        if (!task) {
            cerr << "  Invalid format in " << filename << " line " << lineNum << endl;
            valid = false;
            continue;
        }

        // Проверка уникальности ID
        if (localIds.count(task->id)) {
            cerr << "  Duplicate ID " << task->id << " in " << filename << " line " << lineNum << endl;
            valid = false;
        }
        if (globalIds.count(task->id)) {
            cerr << "  Global duplicate ID " << task->id << " in " << filename << " line " << lineNum << endl;
            valid = false;
        }

        localIds.insert(task->id);
        globalIds.insert(task->id);

        // Проверка результата
        double expected = calculateExpected(*task, opType);
        if (!validateResult(expected, task->result)) {
            cerr << scientific << setprecision(4)
                 << "  Invalid result in " << filename << " line " << lineNum
                 << " Expected: " << expected << " Got: " << task->result << endl;
            valid = false;
        }
    }

    return valid;
}

int main() {
    vector<string> files = {"results_pow.txt", "results_sin.txt", "results_sqrt.txt"};
    set<size_t> globalIds;
    bool allValid = true;

    cout << "Running tests:" << endl;
    for (const auto& file : files) {
        cout << " Cheking \"" + file + "\"" << endl;
        if (!processFile(file, globalIds)) {
            allValid = false;
            continue;
        }
    }

    cout << (allValid ? "All files are valid" : "Validation failed") << endl;
    return allValid ? 0 : 1;
}
