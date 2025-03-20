#include <iostream>
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <atomic>
#include <random>
#include <fstream>
#include <cmath>
#include <variant>
#include <iomanip>

#define FILLW 8

// Структуры для хранения аргументов задач
struct OneArgTask {
    double arg;
};

struct TwoArgsTask {
    float arg1;
    float arg2;
};

using TaskArgs = std::variant<OneArgTask, TwoArgsTask>;

struct TaskInfo {
    size_t id;
    TaskArgs args;
};

// Шаблон сервера
template<typename T>
class Server {
public:
    Server(size_t num_threads = std::thread::hardware_concurrency()) {
        workers.reserve(num_threads);
        start();
    }

    ~Server() {
        stop();
    }

    void start() {
        running = true;
        for (size_t i = 0; i < workers.capacity(); ++i) {
            workers.emplace_back([this] { worker_loop(); });
        }
    }

    void stop() {
        running = false;
        cv.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    size_t add_task(std::function<T()> task) {
        size_t id = next_id++;
        auto promise = std::make_shared<std::promise<T>>();
        std::shared_future<T> future = promise->get_future().share();

        {
            std::lock_guard<std::mutex> lock(results_mutex);
            results[id] = future;
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.emplace([task = std::move(task), promise = std::move(promise)]() mutable {
                try {
                    T result = task();
                    promise->set_value(result);
                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            });
        }

        cv.notify_one();
        return id;
    }

    T request_result(size_t id) {
        std::shared_future<T> future;
        {
            std::lock_guard<std::mutex> lock(results_mutex);
            auto it = results.find(id);
            if (it == results.end()) {
                throw std::runtime_error("Task ID not found");
            }
            future = it->second;
        }
        return future.get();
    }

private:
    void worker_loop() {
        while (running) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this] { return !tasks.empty() || !running; });
                if (!running) break;
                if (!tasks.empty()) {
                    task = std::move(tasks.front());
                    tasks.pop();
                }
            }
            if (task) task();
        }
    }

    std::atomic<bool> running{false};
    std::atomic<size_t> next_id{1};
    std::queue<std::function<void()>> tasks;
    std::unordered_map<size_t, std::shared_future<T>> results;
    std::mutex queue_mutex;
    std::mutex results_mutex;
    std::condition_variable cv;
    std::vector<std::thread> workers;
};

// Клиентские функции с сохранением аргументов
void client_sin(Server<double>& server, int N, const std::string& filename) {
    std::vector<TaskInfo> tasks;
    tasks.reserve(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 2 * M_PI);

    for (int i = 0; i < N; ++i) {
        double arg = dis(gen);
        size_t id = server.add_task([arg] { return std::sin(arg); });
        tasks.push_back({id, OneArgTask{arg}});
    }

    std::ofstream file(filename);
    for (const auto& task : tasks) {
        double result = server.request_result(task.id);
        const auto& args = std::get<OneArgTask>(task.args);
        file << "[" << std::setw(FILLW) << std::setfill('0') << task.id << "] (" << args.arg << ") => " << result << "\n";
    }
}

void client_sqrt(Server<double>& server, int N, const std::string& filename) {
    std::vector<TaskInfo> tasks;
    tasks.reserve(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    for (int i = 0; i < N; ++i) {
        double arg = dis(gen);
        size_t id = server.add_task([arg] { return std::sqrt(arg); });
        tasks.push_back({id, OneArgTask{arg}});
    }

    std::ofstream file(filename);
    for (const auto& task : tasks) {
        double result = server.request_result(task.id);
        const auto& args = std::get<OneArgTask>(task.args);
        file << "[" << std::setw(FILLW) << std::setfill('0') << task.id << "] (" << args.arg << ") => " << result << "\n";
    }
}

void client_pow(Server<double>& server, int N, const std::string& filename) {
    std::vector<TaskInfo> tasks;
    tasks.reserve(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 10.0);

    for (int i = 0; i < N; ++i) {
        float x = static_cast<float>(dis(gen));
        float y = static_cast<float>(dis(gen));
        size_t id = server.add_task([x, y] { return std::pow(x, y); });
        tasks.push_back({id, TwoArgsTask{x, y}});
    }

    std::ofstream file(filename);
    for (const auto& task : tasks) {
        double result = server.request_result(task.id);
        const auto& args = std::get<TwoArgsTask>(task.args);
        file << "[" << std::setw(FILLW) << std::setfill('0') << task.id << "] (" << args.arg1 << ", " << args.arg2 << ") => " << result << "\n";
    }
}

int main() {
    Server<double> server;

    const int N = 10000;
    std::jthread client1([&server] { client_sin(server, N, "results_sin.txt"); });
    std::jthread client2([&server] { client_sqrt(server, N, "results_sqrt.txt"); });
    std::jthread client3([&server] { client_pow(server, N, "results_pow.txt"); });

    client1.join();
    client2.join();
    client3.join();

    return 0;
}
